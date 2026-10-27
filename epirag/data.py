"""Controlled evidence views over HotpotQA and 2WikiMultihopQA."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi


CONDITIONS = (
    "clean",
    "asymmetry",
    "corruption",
    "temporal",
    "incomplete",
    "collusion",
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def squash(text: str, limit: int = 480) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "."


@dataclass
class Passage:
    pid: str
    title: str
    text: str
    gold: bool
    year: int
    source_id: str

    def prompt_block(self, index: int) -> str:
        return (
            f"[{index}] {self.title} (source={self.source_id}, year={self.year})\n"
            f"{self.text}"
        )


@dataclass
class Instance:
    qid: str
    dataset: str
    question: str
    answer: str
    qtype: str
    passages: list[Passage]
    views: dict[str, list[list[str]]] = field(default_factory=dict)

    def passage_map(self) -> dict[str, Passage]:
        return {p.pid: p for p in self.passages}

    def view_passages(self, condition: str, agent: int) -> list[Passage]:
        mapping = self.passage_map()
        return [mapping[pid] for pid in self.views[condition][agent] if pid in mapping]


def _join_sentences(sentences) -> str:
    if isinstance(sentences, str):
        return sentences
    return " ".join(s.strip() for s in sentences if s and str(s).strip())


def _gold_titles(example: dict) -> set[str]:
    facts = example.get("supporting_facts") or {}
    titles = facts.get("title") or []
    return {str(t) for t in titles}


def _confounder(answer: str, passages: list[Passage]) -> str:
    banned = tokenize(answer)
    for passage in passages:
        if passage.gold:
            continue
        title = passage.title.strip()
        title_toks = tokenize(title)
        if not title or title.lower() == answer.lower():
            continue
        if banned and title_toks == banned:
            continue
        if answer.lower() in title.lower():
            continue
        return title
    return "a different entity"


def _replace_answer(text: str, answer: str, confounder: str) -> str:
    """Plant a contradictory string without splicing short answers into other words."""
    norm = (answer or "").strip()
    tokens = tokenize(norm)
    if norm.lower() in {"yes", "no"} or len(tokens) <= 1:
        filler = "no" if norm.lower() == "yes" else "yes" if norm.lower() == "no" else confounder
        return squash(
            text + f" A later correction states that the answer is {filler}.",
            limit=560,
        )
    pattern = re.compile(r"\b" + re.escape(norm) + r"\b", re.IGNORECASE)
    replaced, count = pattern.subn(confounder, text, count=1)
    if count == 0:
        replaced = text + f" A later correction states that the answer is {confounder}."
    return squash(replaced, limit=560)


def _bm25_order(question: str, passages: list[Passage]) -> list[Passage]:
    if not passages:
        return []
    corpus = [tokenize(p.title + " " + p.text) or ["empty"] for p in passages]
    ranker = BM25Okapi(corpus)
    scores = ranker.get_scores(tokenize(question) or ["empty"])
    order = sorted(range(len(passages)), key=lambda i: float(scores[i]), reverse=True)
    return [passages[i] for i in order]


def _take(passages: list[Passage], k: int) -> list[Passage]:
    seen = set()
    chosen = []
    for passage in passages:
        if passage.pid in seen:
            continue
        seen.add(passage.pid)
        chosen.append(passage)
        if len(chosen) >= k:
            break
    return chosen


def _ids(passages: list[Passage]) -> list[str]:
    return [p.pid for p in passages]


def build_instance(example: dict, dataset: str, view_k: int = 3) -> Instance:
    gold = _gold_titles(example)
    context = example["context"]
    titles = list(context["title"])
    sentences = list(context["sentences"])
    base: list[Passage] = []
    for i, (title, sents) in enumerate(zip(titles, sentences)):
        title = str(title)
        base.append(
            Passage(
                pid=f"p{i}",
                title=title,
                text=squash(_join_sentences(sents)),
                gold=title in gold,
                year=2020,
                source_id=f"s{i}",
            )
        )
    # Drop empty passages but keep at least one.
    base = [p for p in base if p.text] or base
    question = str(example["question"]).strip()
    answer = str(example["answer"]).strip()
    ranked = _bm25_order(question, base)
    golds = [p for p in ranked if p.gold]
    distractors = [p for p in ranked if not p.gold]
    if not golds:
        golds = ranked[:1]
        distractors = ranked[1:]

    confounder = _confounder(answer, base)
    corrupt = Passage(
        pid="corrupt",
        title=golds[0].title,
        text=_replace_answer(golds[0].text, answer, confounder),
        gold=False,
        year=2020,
        source_id="s-corrupt",
    )
    stale = Passage(
        pid="stale",
        title=golds[0].title,
        text=_replace_answer(golds[0].text, answer, confounder),
        gold=False,
        year=2014,
        source_id="s-stale",
    )
    passages = list(base) + [corrupt, stale]

    clean_pool = _take(ranked, view_k)
    clean_views = [_ids(clean_pool) for _ in range(3)]

    g0 = golds[0]
    g1 = golds[1] if len(golds) > 1 else None
    d0 = distractors[:view_k]
    a0 = _take([g0] + d0, view_k)
    if g1 is not None:
        a1 = _take([g1] + distractors[view_k : view_k * 2 + 1], view_k)
    else:
        a1 = _take(distractors[view_k : view_k * 2 + 2], view_k)
    a2 = _take(distractors[view_k * 2 : view_k * 3 + 2] or distractors[:view_k], view_k)
    # Keep the two supporting passages on different agents when both exist.
    if g1 is not None:
        a0 = [p for p in a0 if p.pid != g1.pid]
        a1 = [p for p in a1 if p.pid != g0.pid]
        a2 = [p for p in a2 if p.pid not in {g0.pid, g1.pid}]
        a0 = _take(a0 + d0, view_k)
        a1 = _take(a1 + distractors, view_k)
        a2 = _take(a2 + distractors, view_k)
    asym_views = [_ids(a0), _ids(a1), _ids(a2)]

    corrupt_views = [_ids(a0), _ids(a1), _ids(_take([corrupt] + a2[1:], view_k))]
    # Temporal views use the same split, but the clean supporting source is dated
    # and one agent receives the stale copy instead of an unrelated distractor.
    temporal_views = [_ids(a0), _ids(a1), _ids(_take([stale] + a2[1:], view_k))]

    withheld = g1.pid if g1 is not None else g0.pid
    reduced = [p for p in ranked if p.pid != withheld]
    incomplete_pool = _take(reduced, view_k)
    incomplete_views = [_ids(incomplete_pool) for _ in range(3)]

    # Two agents share the corrupted page and never see a supporting page.
    # The third agent keeps one supporting page. This is a constructed
    # colluding majority, not a naturally occurring web conflict.
    majority_pool = [p for p in distractors if p.pid != corrupt.pid]
    majority_a = _take([corrupt] + majority_pool, view_k)
    majority_b = _take([corrupt] + majority_pool[view_k:], view_k)
    collusion_views = [_ids(majority_a), _ids(majority_b), _ids(a0)]

    instance = Instance(
        qid=str(example.get("id", question[:40])),
        dataset=dataset,
        question=question,
        answer=answer,
        qtype=str(example.get("type", "unknown")),
        passages=passages,
        views={
            "clean": clean_views,
            "asymmetry": asym_views,
            "corruption": corrupt_views,
            "temporal": temporal_views,
            "incomplete": incomplete_views,
            "collusion": collusion_views,
        },
    )
    _apply_years(instance)
    return instance


def _apply_years(instance: Instance) -> None:
    """Timestamps differ only in the temporal condition's copies.

    The method reads the year stored on the passage object. Non-temporal
    conditions must not observe a recency cue, so those views only contain
    passages whose year remains the default.
    """
    for passage in instance.passages:
        if passage.pid == "stale":
            passage.year = 2014
        else:
            passage.year = 2020
    # Date the clean supporting passages only when they appear in the temporal
    # view. Other conditions keep the default year on the shared objects, so
    # temporal dating is applied by cloning at prompt time in the runner.


def temporal_passage_copy(passage: Passage) -> Passage:
    year = 2023 if passage.gold else passage.year
    if passage.pid == "stale":
        year = 2014
    return Passage(
        pid=passage.pid,
        title=passage.title,
        text=passage.text,
        gold=passage.gold,
        year=year,
        source_id=passage.source_id,
    )


def select_examples(dataset, dataset_name: str, n_eval: int, n_dev: int, seed: int, block: str = "eval"):
    del dataset_name
    by_type: dict[str, list[int]] = {}
    for i, row in enumerate(dataset):
        by_type.setdefault(str(row.get("type", "unknown")), []).append(i)
    rng = random.Random(seed)
    for ids in by_type.values():
        rng.shuffle(ids)
    order = ["bridge", "compositional", "comparison", "inference", "unknown"]
    buckets = [list(by_type[key]) for key in order if by_type.get(key)]
    known = {key for key in order}
    extra = []
    for key, ids in by_type.items():
        if key not in known:
            extra.extend(ids)
    if extra:
        rng.shuffle(extra)
        buckets.append(extra)
    mixed: list[int] = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                mixed.append(bucket.pop(0))
    dev_ids = mixed[:n_dev]
    eval_ids = mixed[n_dev : n_dev + n_eval]
    # The confirmation block starts after the reported development slice.
    # It is not used to choose thresholds or decision rules.
    confirm_ids = mixed[n_dev + n_eval : n_dev + 2 * n_eval]
    locked_ids = mixed[n_dev + 2 * n_eval : n_dev + 3 * n_eval]
    final_ids = mixed[n_dev + 3 * n_eval : n_dev + 4 * n_eval]
    holdout_ids = mixed[n_dev + 4 * n_eval : n_dev + 5 * n_eval]
    advance_ids = mixed[n_dev + 5 * n_eval : n_dev + 6 * n_eval]
    forward_ids = mixed[n_dev + 6 * n_eval : n_dev + 7 * n_eval]
    prospect_ids = mixed[n_dev + 7 * n_eval : n_dev + 8 * n_eval]
    ledger_ids = mixed[n_dev + 8 * n_eval : n_dev + 9 * n_eval]
    # Folio is the next disjoint block. It is reserved for the derivation test.
    folio_ids = mixed[n_dev + 9 * n_eval : n_dev + 10 * n_eval]
    if block == "confirm":
        return dev_ids, confirm_ids
    if block == "locked":
        return dev_ids, locked_ids
    if block == "final":
        return dev_ids, final_ids
    if block == "holdout":
        return dev_ids, holdout_ids
    if block == "advance":
        return dev_ids, advance_ids
    if block == "forward":
        return dev_ids, forward_ids
    if block == "prospect":
        return dev_ids, prospect_ids
    if block == "ledger":
        return dev_ids, ledger_ids
    if block == "folio":
        return dev_ids, folio_ids
    if block == "dev":
        return dev_ids, dev_ids
    return dev_ids, eval_ids
