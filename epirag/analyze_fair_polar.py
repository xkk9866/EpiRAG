"""Apply the same polar binding to every method. Fair comparison."""

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
    by.setdefault(row["method"], []).append(row)

methods = [
    "independent",
    "centralized",
    "debate",
    "collab",
    "vote",
    "epirag",
    "epirag_flat",
    "epirag_cover",
    "epirag_commit",
    "epirag_revise",
]
for method in methods:
    group = by[method]
    base_f1 = 100 * sum(row["f1"] for row in group) / len(group)
    base_em = 100 * sum(row["em"] for row in group) / len(group)
    pol_f1 = 100 * sum(
        token_f1(canonicalize_polar(row["question"], row["answer"]), row["gold"]) for row in group
    ) / len(group)
    pol_em = 100 * sum(
        exact_match(canonicalize_polar(row["question"], row["answer"]), row["gold"]) for row in group
    ) / len(group)
    changed = sum(
        canonicalize_polar(row["question"], row["answer"]).lower() != (row["answer"] or "").strip().lower()
        for row in group
    )
    print(
        f"{method:22} base EM {base_em:5.1f} F1 {base_f1:5.1f} | "
        f"polar EM {pol_em:5.1f} F1 {pol_f1:5.1f} changed {changed}"
    )
