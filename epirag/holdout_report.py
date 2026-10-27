"""Held-out block. Families fixed before this block is scored.

Family A, one reader, Holm over 10 tests. Stress-pooled exact match.
Stress is asymmetry, corruption, temporal, and incomplete.
Contrasts: epirag_gate against debate, centralized, epirag_flat,
epirag_graph, and epirag_cover, on each dataset.

Family B is the same 10 tests on the 3B reader, Holm inside B only.

Thresholds are the frozen contradiction filter (0.55 / 0.25) and the
frozen support floor (0.30) and margin (0.10). They are not fit here.
"""

from pathlib import Path

from epirag.locked_report import CELLS, DATASETS, STRESS, fair_rows, family, holm, pooled
from epirag.paired_tests import load

ROOT = Path(__file__).resolve().parents[1]
FAMILY_A = [
    ("epirag_gate", "debate"),
    ("epirag_gate", "centralized"),
    ("epirag_gate", "epirag_flat"),
    ("epirag_gate", "epirag_graph"),
    ("epirag_gate", "epirag_cover"),
]


def branch_rates(rows, method: str = "epirag_gate"):
    print(f"\n-- {method} decisions")
    from collections import defaultdict

    buckets = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["method"] != method:
            continue
        buckets[(row["dataset"], row["condition"])][row.get("decision") or ""] += 1
    for key in sorted(buckets):
        total = sum(buckets[key].values())
        parts = [f"{name} {count}/{total}" for name, count in sorted(buckets[key].items())]
        print(key, " ".join(parts))


def report(path: Path):
    rows = load(path)
    print(f"\n##### {path} rows {len(rows)}")
    grouped = fair_rows(rows)
    from epirag.locked_report import cells, cost_table

    cells(
        grouped,
        [
            "centralized",
            "debate",
            "epirag_cover",
            "epirag_flat",
            "epirag_graph",
            "epirag_gate",
        ],
    )
    family(grouped, FAMILY_A, STRESS, "family A/B stress")
    branch_rates(rows)
    cost_table(grouped)


if __name__ == "__main__":
    report(ROOT / "experiments" / "results" / "records_holdout.jsonl")
    other = ROOT / "experiments" / "results" / "records_holdout_3b.jsonl"
    if other.exists() and other.stat().st_size:
        report(other)
