"""Stress means on the confirmation slice, one decimal, fair scores."""

from pathlib import Path

import numpy as np

from epirag.paired_tests import CONDITIONS, STRESS, load
from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, token_f1

rows = load(Path("experiments/results/records_confirm.jsonl"))
methods = [
    "centralized",
    "debate",
    "collab",
    "epirag_cover",
    "epirag_flat",
    "epirag_drop",
    "epirag_cover_noret",
]
for dataset in ("hotpotqa", "2wiki"):
    for method in methods:
        f1s = []
        toks = []
        for condition in STRESS:
            items = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["condition"] == condition
                and row["method"] == method
            ]
            scores = []
            for row in items:
                pred = canonicalize_polar(row["question"], row["answer"])
                scores.append(token_f1(pred, row["gold"]))
            f1s.append(float(np.mean(scores)))
            toks.append(float(np.mean([row["comm_tokens"] for row in items])))
        print(
            f"{dataset:9} {method:22} F1 {100 * float(np.mean(f1s)):.1f} "
            f"tok {float(np.mean(toks)):.0f}"
        )
