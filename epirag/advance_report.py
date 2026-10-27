"""Advance block. Family fixed before this block is scored.

Family A, one reader, Holm over 10 tests. Stress-pooled exact match.
Stress is asymmetry, corruption, temporal, and incomplete.
Contrasts: epirag_mark against debate, centralized, epirag_flat,
epirag_graph, and epirag_cover, on each dataset.

epirag_mark keeps the BM25 page pool. It labels sentences only when a
cross-source contradiction of the leading claim is present. The cutoffs
are the frozen filter (contradiction above 0.55, entailment below 0.25,
support entailment 0.40) and the frozen lookup trigger (margin 0.10,
support floor 0.30). They are not fit on this block.
"""

from pathlib import Path

from epirag.holdout_report import branch_rates
from epirag.locked_report import cost_table, cells, fair_rows
from epirag.paired_tests import load

ROOT = Path(__file__).resolve().parents[1]
FAMILY_A = [
    ("epirag_mark", "debate"),
    ("epirag_mark", "centralized"),
    ("epirag_mark", "epirag_flat"),
    ("epirag_mark", "epirag_graph"),
    ("epirag_mark", "epirag_cover"),
]
METHODS = [
    "centralized",
    "debate",
    "epirag_cover",
    "epirag_flat",
    "epirag_graph",
    "epirag_mark",
]


def report(path: Path):
    rows = load(path)
    print(f"\n##### {path} rows {len(rows)}")
    grouped = fair_rows(rows)
    from epirag.locked_report import STRESS, family

    cells(grouped, METHODS)
    family(grouped, FAMILY_A, STRESS, "family A stress")
    branch_rates(rows, "epirag_mark")
    cost_table(grouped)


if __name__ == "__main__":
    report(ROOT / "experiments" / "results" / "records_advance.jsonl")
