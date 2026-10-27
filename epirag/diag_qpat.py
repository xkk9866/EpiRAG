"""Classify comparison questions. Development only; does not choose a cutoff."""

import json
import re
from collections import defaultdict

from epirag.engine import canonicalize_polar
from epirag.execute import plan
from epirag.metrics import exact_match

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
PATH = "experiments/results/records_prospect.jsonl"


def bucket(question: str) -> str:
    if plan(question):
        return "year-compare"
    text = question.lower()
    if re.search(r"\b(same|both|in common|share)\b", text):
        return "same-attribute"
    if " or " in text:
        return "other-or"
    return "other"


def main():
    rows = []
    with open(PATH, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS or row["method"] not in {"epirag_cover", "epirag_exec", "centralized"}:
                continue
            rows.append(row)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    samples = defaultdict(list)
    for methods in grouped.values():
        cover = methods["epirag_cover"]
        kind = bucket(cover["question"])
        key = (cover["dataset"], cover["qtype"], kind)
        stats[key][0][0] += 1
        stats[key][0][1] += exact_match(canonicalize_polar(cover["question"], cover["answer"]), cover["gold"])
        stats[key][0][2] += exact_match(canonicalize_polar(cover["question"], methods["epirag_exec"]["answer"]), cover["gold"])
        if len(samples[key]) < 3:
            samples[key].append(cover["question"][:140])
    for key in sorted(stats):
        n, cover_hit, exec_hit = stats[key][0]
        print(f"{key[0]:10} {key[1]:20} {key[2]:16} n {n:4} cover {100 * cover_hit / n:5.1f} exec {100 * exec_hit / n:5.1f}")
        for question in samples[key]:
            print("   ", question)


if __name__ == "__main__":
    main()
