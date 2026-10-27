"""Development: exact match of a cross-view chain, else the stored witness answer.

No threshold is chosen from these scores. Abstention falls back to cover.
"""

import json
import re
from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.diag_frontier import anchors, contains_anchor, contains_mention, mentions
from epirag.engine import QUESTION_STOP, canonicalize_polar, split_sentences
from epirag.metrics import exact_match, normalize_answer
from epirag.run_eval import materialize

STRESS = ("asymmetry", "corruption", "temporal", "incomplete")
FILES = {
    "advance": "experiments/results/records_advance.jsonl",
    "forward": "experiments/results/records_forward.jsonl",
}
DATE = re.compile(r"\b(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b")
YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")


def content_words(question: str) -> set[str]:
    blocked = set(QUESTION_STOP) | set(anchors(question))
    return {token for token in normalize_answer(question).split() if len(token) > 3 and token not in blocked}


def sentences_of(pages):
    rows = []
    for passage in pages:
        pieces = split_sentences(passage.text) or [passage.text[:240]]
        for text in pieces:
            rows.append((text, passage.source_id))
    return rows


def candidates_in(text: str, question: str, question_anchors: list[str]) -> list[str]:
    found = []
    for mention in mentions(text):
        if mention in question_anchors:
            continue
        if mention not in found:
            found.append(mention)
    want_dates = bool(re.search(r"\b(when|date|year|born|birthday)\b", question.lower()))
    for match in DATE.findall(text):
        key = match.lower()
        if key not in found:
            found.append(key)
    if want_dates:
        for match in YEAR.findall(text):
            if match not in found:
                found.append(match)
    return found


def chain_scored(question: str, pages):
    question_anchors = anchors(question)
    words = content_words(question)
    rows = sentences_of(pages)
    if not question_anchors or not rows:
        return ""
    hop0 = [(text, source) for text, source in rows if contains_anchor(text, question_anchors)]
    introduced = []
    for text, _source in hop0:
        for mention in mentions(text):
            if mention in question_anchors or mention in introduced:
                continue
            introduced.append(mention)
    scored = {}
    for text, source in rows:
        cands = candidates_in(text, question, question_anchors)
        if not cands:
            continue
        have = set(normalize_answer(text).split())
        relation = sum(token in have for token in words)
        linked = contains_anchor(text, question_anchors)
        bridge = False
        if not linked:
            bridge = any(contains_mention(text, mention) for mention in introduced)
        if not linked and not bridge:
            continue
        # A hub that only repeats the question entity is not an endpoint.
        hop = 0 if linked else 1
        for cand in cands:
            # Prefer endpoints on the second hop, then a sentence that uses the question's own words.
            rank = (hop, relation, -len(normalize_answer(cand).split()))
            previous = scored.get(cand)
            if previous is None or rank > previous[0]:
                scored[cand] = (rank, source)
    return scored


def chain_answer(question: str, pages):
    scored = chain_scored(question, pages)
    if not scored:
        return ""
    best_rank = max(item[0] for item in scored.values())
    winners = [cand for cand, (rank, _source) in scored.items() if rank == best_rank]
    if len(winners) != 1:
        return ""
    return winners[0]


def pages_of(instance, condition):
    agents, _archive = materialize(instance, condition)
    pages = []
    seen = set()
    for group in agents:
        for passage in group:
            if passage.pid in seen:
                continue
            seen.add(passage.pid)
            pages.append(passage)
    return pages


def main():
    stored = defaultdict(dict)
    for split, path in FILES.items():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["condition"] not in STRESS or row["method"] != "epirag_cover":
                    continue
                stored[split][(row["dataset"], row["qid"], row["condition"])] = row
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    specs = (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160))
    for split in FILES:
        print("==", split, flush=True)
        for name, data, budget in specs:
            _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block=split)
            instances = [build_instance(data[i], name) for i in ids]
            cover_hit = hybrid_hit = fired = fire_hit = n = 0
            for instance in instances:
                for condition in STRESS:
                    row = stored[split].get((name, instance.qid, condition))
                    if row is None:
                        continue
                    n += 1
                    cover = canonicalize_polar(instance.question, row["answer"])
                    structural = chain_answer(instance.question, pages_of(instance, condition))
                    pred = structural or cover
                    if structural:
                        fired += 1
                        fire_hit += exact_match(structural, instance.answer)
                    cover_hit += exact_match(cover, instance.answer)
                    hybrid_hit += exact_match(pred, instance.answer)
            print(
                f"{name} n {n} cover {100 * cover_hit / n:.2f} hybrid {100 * hybrid_hit / n:.2f} "
                f"fire {100 * fired / n:.1f} precision {100 * fire_hit / fired if fired else 0:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
