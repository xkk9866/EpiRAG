"""Development-only: does a question-anchored frontier still contain the answer?

Scores no reader. A high stated-rate with a smaller mention set is the
reason to spend a fresh slice on the rule. This file does not choose
thresholds from exact match.
"""

from __future__ import annotations

import re
from collections import Counter

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import QUESTION_STOP, Snippet, _rank_sentences, split_sentences
from epirag.metrics import normalize_answer
from epirag.run_eval import materialize

STRESS = ("asymmetry", "corruption", "temporal", "incomplete")
CAP_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
INITIAL = {"The", "A", "An", "This", "That", "These", "Those", "It", "He", "She", "They", "His", "Her"}


def anchors(question: str) -> list[str]:
    phrases = []
    for match in CAP_PHRASE.findall(question):
        if match in INITIAL:
            continue
        key = match.lower()
        if key not in phrases:
            phrases.append(key)
    if phrases:
        return phrases
    words = []
    for token in re.findall(r"[a-z0-9]+", question.lower()):
        if len(token) <= 3 or token in QUESTION_STOP or token in words:
            continue
        words.append(token)
    return words[:8]


def mentions(text: str) -> list[str]:
    found = []
    stripped = text.lstrip()
    initial = stripped.split(" ", 1)[0] if stripped else ""
    for match in CAP_PHRASE.findall(text):
        if match in INITIAL:
            continue
        if match == initial and " " not in match:
            continue
        key = match.lower()
        if key not in found:
            found.append(key)
    return found


def contains_anchor(text: str, question_anchors: list[str]) -> bool:
    lowered = f" {text.lower()} "
    for anchor in question_anchors:
        if f" {anchor} " in lowered:
            return True
    return False


def contains_mention(text: str, mention: str) -> bool:
    return f" {mention} " in f" {text.lower()} "


def frontier(question: str, passages) -> list[Snippet]:
    question_anchors = anchors(question)
    snippets = []
    for passage in passages:
        for sentence in split_sentences(passage.text):
            snippets.append(Snippet(text=sentence, passage=passage, rho=1.0))
    if not snippets or not question_anchors:
        return snippets[:4]
    hop0 = [snippet for snippet in snippets if contains_anchor(snippet.text, question_anchors)]
    introduced = []
    for snippet in hop0:
        for mention in mentions(snippet.text):
            if mention in question_anchors or mention in introduced:
                continue
            introduced.append(mention)
    hop1 = []
    for snippet in snippets:
        if snippet in hop0:
            continue
        if any(contains_mention(snippet.text, mention) for mention in introduced):
            hop1.append(snippet)
    # Prefer a hop-1 sentence from a different source than the sentence that
    # introduced its mention. Same-source sentences fill only the leftover slots.
    cross = []
    same = []
    hop0_sources = {}
    for snippet in hop0:
        for mention in mentions(snippet.text):
            hop0_sources.setdefault(mention, set()).add(snippet.passage.source_id)
    for snippet in hop1:
        crossed = False
        for mention in mentions(snippet.text):
            sources = hop0_sources.get(mention, set())
            if sources and snippet.passage.source_id not in sources:
                crossed = True
                break
        (cross if crossed else same).append(snippet)
    chosen = hop0[:2] + cross[:2]
    if len(chosen) < 4:
        for snippet in same:
            if len(chosen) >= 4:
                break
            chosen.append(snippet)
    if len(chosen) < 4:
        for snippet in hop0[2:]:
            if len(chosen) >= 4:
                break
            chosen.append(snippet)
    return chosen[:4] or snippets[:4]


def stated(text: str, answer: str) -> bool:
    gold = normalize_answer(answer).split()
    if not gold:
        return False
    have = set(normalize_answer(text).split())
    return sum(token in have for token in gold) / len(gold) >= 0.5


def main():
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    specs = (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160))
    for name, data, budget in specs:
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="advance")
        instances = [build_instance(data[i], name) for i in ids]
        for label, picker in (("frontier", frontier), ("bm25", lambda q, pages: _rank_sentences(q, pages)[:4])):
            hits = 0
            total = 0
            mention_counts = []
            unique_hits = 0
            unique_tries = 0
            by_type = Counter()
            by_type_n = Counter()
            for instance in instances:
                for condition in STRESS:
                    agents, _archive = materialize(instance, condition)
                    pages = []
                    seen = set()
                    for group in agents:
                        for passage in group:
                            if passage.pid in seen:
                                continue
                            seen.add(passage.pid)
                            pages.append(passage)
                    chosen = picker(instance.question, pages)
                    text = " ".join(snippet.text for snippet in chosen)
                    ok = stated(text, instance.answer)
                    hits += int(ok)
                    total += 1
                    by_type[instance.qtype] += int(ok)
                    by_type_n[instance.qtype] += 1
                    mention_counts.append(len(mentions(text)))
                    if label == "frontier":
                        question_anchors = set(anchors(instance.question))
                        cands = [m for m in mentions(text) if m not in question_anchors]
                        if len(cands) == 1:
                            unique_tries += 1
                            unique_hits += int(normalize_answer(cands[0]) == normalize_answer(instance.answer))
            rate = 100 * hits / total
            mean_mentions = sum(mention_counts) / len(mention_counts)
            print(f"{name} {label} stated {rate:.1f} n {total} mentions {mean_mentions:.2f}", flush=True)
            if label == "frontier":
                prec = 100 * unique_hits / unique_tries if unique_tries else 0.0
                print(
                    f"  unique-mention tries {unique_tries} precision {prec:.1f} "
                    f"em-if-only-those {100 * unique_hits / total:.1f}",
                    flush=True,
                )
            for qtype, count in sorted(by_type_n.items()):
                print(f"  {qtype} {100 * by_type[qtype] / count:.1f}", flush=True)


if __name__ == "__main__":
    main()
