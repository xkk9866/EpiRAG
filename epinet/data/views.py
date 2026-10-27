"""Private evidence views and controlled stress conditions.

Each of ``n_agents`` agents receives ``k`` pages. Conditions:

clean       every agent holds the same top-k BM25 pages;
asymmetry   supporting pages are dealt round-robin to different agents;
corruption  asymmetry, plus a rewritten copy of the answer page, whose answer
            span is replaced by a type-consistent confounder, in a view that
            does not hold the answer page;
temporal    corruption, but the rewritten copy is dated 2014 and supporting
            pages are dated 2023 (a stale copy with a recency cue);
incomplete  asymmetry with one non-answer supporting page withheld from all
            views (the derivation cannot be completed from the views);
collusion   two agents hold the same corrupted copy, the third holds the
            supporting pages (a correlated, wrong majority).

The confounder never depends on any method; all methods see the same views.
"""

from __future__ import annotations

import math
import re

from rank_bm25 import BM25Okapi

from epinet.data.examples import Example, Page
from epinet.text import normalize_answer, parse_value, tokenize

CONDITIONS = ("clean", "asymmetry", "corruption", "temporal", "incomplete", "collusion")
STRESS = ("asymmetry", "corruption", "temporal", "incomplete")


def bm25_order(query: str, pages: list[Page]) -> list[Page]:
    if len(pages) <= 1:
        return list(pages)
    corpus = [tokenize(p.title + " " + p.text) or ["empty"] for p in pages]
    scores = BM25Okapi(corpus).get_scores(tokenize(query) or ["empty"])
    order = sorted(range(len(pages)), key=lambda i: (-float(scores[i]), i))
    return [pages[i] for i in order]


def _contains(text: str, answer: str) -> bool:
    a = normalize_answer(answer)
    return bool(a) and f" {a} " in f" {normalize_answer(text)} "


def answer_page(example: Example, golds: list[Page]) -> Page:
    for page in golds:
        if any(_contains(page.text, ans) for ans in example.answers):
            return page
    return golds[-1]


def confounder(example: Example, distractors: list[Page]) -> str:
    """A wrong answer of the same surface type as the gold answer."""
    answer = example.answer
    key = normalize_answer(answer)
    if key in {"yes", "no"}:
        return "no" if key == "yes" else "yes"
    value = parse_value(answer)
    if not math.isnan(value) and re.search(r"\d", answer):
        digits = re.findall(r"\d+", answer)
        shift = 3 + (sum(int(d) for d in digits) % 11)
        target = max(digits, key=len)
        return answer.replace(target, str(int(target) + shift if int(target) < 1990 else int(target) - shift), 1)
    n_tok = max(1, len(tokenize(answer)))
    best = None
    for page in distractors:
        title = re.sub(r"\s*\(.*?\)\s*", " ", page.title).strip()
        if not title or normalize_answer(title) == key or key in normalize_answer(title):
            continue
        gap = abs(len(tokenize(title)) - n_tok)
        if best is None or gap < best[0]:
            best = (gap, title)
    return best[1] if best else "a different entity"


def corrupt_copy(page: Page, example: Example, wrong: str, pid: str, kind: str, year: int) -> Page:
    sentences = list(page.sentences)
    replaced = False
    for ans in example.answers:
        pattern = re.compile(r"\b" + re.escape(ans.strip()) + r"\b", re.IGNORECASE)
        for i, sentence in enumerate(sentences):
            new, count = pattern.subn(wrong, sentence, count=1)
            if count:
                sentences[i] = new
                replaced = True
                break
        if replaced:
            break
    if not replaced:
        sentences.append(f"According to a later revision, the answer is {wrong}.")
    return Page(pid=pid, title=page.title, sentences=sentences, gold=False, support=[], year=year, kind=kind)


def _fill(chosen: list[Page], pool: list[Page], k: int, banned: set[str]) -> list[Page]:
    out = list(chosen)
    have = {p.pid for p in out} | banned
    for page in pool:
        if len(out) >= k:
            break
        if page.pid in have:
            continue
        out.append(page)
        have.add(page.pid)
    return out[:k]


def build_views(example: Example, condition: str, n_agents: int = 3, k: int = 3) -> list[list[Page]]:
    pages = [p.copy() for p in example.pages]
    ranked = bm25_order(example.question, pages)
    golds = [p for p in ranked if p.gold] or ranked[:1]
    distractors = [p for p in ranked if p.pid not in {g.pid for g in golds}]
    ans_page = answer_page(example, golds)
    wrong = confounder(example, distractors)

    if condition == "clean":
        top = ranked[:k]
        return [[p.copy() for p in top] for _ in range(n_agents)]

    def split_golds(gold_list: list[Page]) -> list[list[Page]]:
        dealt: list[list[Page]] = [[] for _ in range(n_agents)]
        for i, page in enumerate(gold_list):
            dealt[i % n_agents].append(page)
        return dealt

    if condition == "incomplete":
        bridges = [g for g in golds if g.pid != ans_page.pid]
        withheld = bridges[0].pid if bridges else None
        kept = [g for g in golds if g.pid != withheld]
        dealt = split_golds(kept)
        banned = {withheld} if withheld else set()
        views = []
        for i in range(n_agents):
            views.append(_fill(dealt[i], distractors[i * k :] + distractors, k, banned))
        return views

    dealt = split_golds(golds)
    views = []
    for i in range(n_agents):
        views.append(_fill(dealt[i], distractors[i * k :] + distractors, k, set()))

    if condition == "asymmetry":
        return views

    holder = next((i for i, v in enumerate(views) if any(p.pid == ans_page.pid for p in v)), 0)
    if condition in {"corruption", "temporal"}:
        target = next(
            (i for i in range(n_agents) if i != holder and not any(p.gold for p in views[i])),
            (holder + 1) % n_agents,
        )
        if condition == "corruption":
            bad = corrupt_copy(ans_page, example, wrong, "corrupt", "corrupt", 2020)
        else:
            bad = corrupt_copy(ans_page, example, wrong, "stale", "stale", 2014)
        slots = [j for j in range(len(views[target])) if not views[target][j].gold]
        if slots:
            views[target][slots[-1]] = bad
        else:
            views[target].append(bad)
        if condition == "temporal":
            for view in views:
                for page in view:
                    if page.gold:
                        page.year = 2023
        return views

    if condition == "collusion":
        bad = corrupt_copy(ans_page, example, wrong, "corrupt", "corrupt", 2020)
        honest = _fill(golds[:k], distractors, k, set())
        majority_a = _fill([bad], distractors, k, set())
        majority_b = _fill([bad.copy()], distractors[k - 1 :] + distractors, k, set())
        return [majority_a, majority_b, honest][:n_agents] if n_agents >= 3 else [majority_a, honest]

    raise ValueError(condition)
