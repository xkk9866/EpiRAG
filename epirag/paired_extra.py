"""Extra paired tests for reliability and the shortened role filter."""

from epirag.paired_tests import load, paired
from pathlib import Path

rows = load(Path("experiments/results/records_v2.jsonl"))
pairs = [
    ("epirag_cover", "epirag_cover_norel"),
    ("epirag_cover", "epirag"),
    ("epirag_drop", "epirag"),
    ("epirag_cover", "debate"),
]
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        for left, right in pairs:
            stats = paired(rows, left, right, dataset, condition)
            if stats is None or stats["p"] > 0.1 and abs(stats["em_a"] - stats["em_b"]) < 5:
                continue
            print(
                f"{dataset:9} {condition:11} {left:22} vs {right:22} "
                f"EM {stats['em_a']:.1f}/{stats['em_b']:.1f} "
                f"F1 {stats['f1_a']:.1f}/{stats['f1_b']:.1f} "
                f"only {stats['a_only']}/{stats['b_only']} p={stats['p']:.4f}"
            )

triggered = {"hotpotqa": {}, "2wiki": {}}
for row in rows:
    if row["method"] != "epirag_cover":
        continue
    triggered[row["dataset"]].setdefault(row["condition"], []).append(bool(row["triggered"]))
for dataset, conds in triggered.items():
    for condition, flags in conds.items():
        rate = 100 * sum(flags) / len(flags)
        print(f"trigger {dataset} {condition} {rate:.1f} n={len(flags)}")
