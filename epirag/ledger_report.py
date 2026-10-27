"""Ledger block. Family fixed before this block is scored.

Family A, one reader, Holm over 10 tests. Stress-pooled exact match.
Stress is asymmetry, corruption, temporal, and incomplete.
Contrasts: epirag_hop against debate, centralized, epirag_flat,
epirag_graph, and epirag_cover, on each dataset.

epirag_hop keeps the witness-bundle reader, except in two frozen cases.
A two-entity year comparison is decided by the two years, outside the reader.
A relational question is split into a bridge hop and a property hop, and the
property span is kept only when it is stated in that hop's sentences.
No cutoff is fit on this block.
"""

from pathlib import Path

from epirag.holdout_report import branch_rates
from epirag.locked_report import STRESS, cells, cost_table, fair_rows, family
from epirag.paired_tests import load

ROOT = Path(__file__).resolve().parents[1]
FAMILY_A = [
    ("epirag_hop", "debate"),
    ("epirag_hop", "centralized"),
    ("epirag_hop", "epirag_flat"),
    ("epirag_hop", "epirag_graph"),
    ("epirag_hop", "epirag_cover"),
]
METHODS = [
    "centralized",
    "debate",
    "epirag_cover",
    "epirag_flat",
    "epirag_graph",
    "epirag_hop",
]


def report(path: Path):
    rows = load(path)
    print(f"\n##### {path} rows {len(rows)}")
    grouped = fair_rows(rows)
    cells(grouped, METHODS)
    family(grouped, FAMILY_A, STRESS, "family A stress")
    branch_rates(rows, "epirag_hop")
    cost_table(grouped)


if __name__ == "__main__":
    report(ROOT / "experiments" / "results" / "records_ledger.jsonl")
