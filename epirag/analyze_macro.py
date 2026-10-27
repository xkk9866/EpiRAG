"""Macro comparison of cover, commit, revise, and polar binding."""

import json
from pathlib import Path

from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, token_f1

rows = [
    json.loads(line)
    for line in Path("experiments/results/records_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
by = {}
for row in rows:
    by.setdefault((row["dataset"], row["condition"], row["method"]), []).append(row)

conditions = ("clean", "asymmetry", "corruption", "temporal", "incomplete")


def cell(dataset, condition, method):
    group = by[(dataset, condition, method)]
    em = 100 * sum(row["em"] for row in group) / len(group)
    f1 = 100 * sum(row["f1"] for row in group) / len(group)
    return em, f1


def polar_cell(dataset, condition):
    group = by[(dataset, condition, "epirag_cover")]
    em = 100 * sum(exact_match(canonicalize_polar(row["question"], row["answer"]), row["gold"]) for row in group) / len(group)
    f1 = 100 * sum(token_f1(canonicalize_polar(row["question"], row["answer"]), row["gold"]) for row in group) / len(group)
    return em, f1


methods = {
    "cover": lambda d, c: cell(d, c, "epirag_cover"),
    "commit": lambda d, c: cell(d, c, "epirag_commit"),
    "revise": lambda d, c: cell(d, c, "epirag_revise"),
    "revise_noret": lambda d, c: cell(d, c, "epirag_revise_noret"),
    "polar": polar_cell,
    "debate": lambda d, c: cell(d, c, "debate"),
    "centralized": lambda d, c: cell(d, c, "centralized"),
}
for name, fn in methods.items():
    ems = []
    f1s = []
    print(f"\n{name}")
    for dataset in ("hotpotqa", "2wiki"):
        for condition in conditions:
            em, f1 = fn(dataset, condition)
            ems.append(em)
            f1s.append(f1)
            print(f"  {dataset:10} {condition:12} EM {em:5.1f} F1 {f1:5.1f}")
    print(f"  macro EM {sum(ems)/len(ems):5.1f} F1 {sum(f1s)/len(f1s):5.1f}")
