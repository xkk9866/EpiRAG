"""Per-cell scores after the same yes/no binding for every method."""

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

methods = [
    "independent",
    "centralized",
    "debate",
    "collab",
    "epirag",
    "epirag_flat",
    "epirag_cover",
    "epirag_commit",
    "epirag_revise",
    "epirag_revise_noret",
    "epirag_commit_noret",
]
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        print(f"\n{dataset} {condition}")
        for method in methods:
            group = by[(dataset, condition, method)]
            em = 100 * sum(
                exact_match(canonicalize_polar(r["question"], r["answer"]), r["gold"]) for r in group
            ) / len(group)
            f1 = 100 * sum(
                token_f1(canonicalize_polar(r["question"], r["answer"]), r["gold"]) for r in group
            ) / len(group)
            tok = sum(r["comm_tokens"] for r in group) / len(group)
            print(f"  {method:22} EM {em:5.1f} F1 {f1:5.1f} tok {tok:6.1f}")
