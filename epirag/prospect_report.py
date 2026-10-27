"""Prospect block. Family fixed before this block is scored.

Family A, one reader, Holm over 10 tests. Stress-pooled exact match.
Stress is asymmetry, corruption, temporal, and incomplete.
Contrasts: epirag_exec against debate, centralized, epirag_flat,
epirag_graph, and epirag_cover, on each dataset.

epirag_exec answers a two-entity comparison by reading one year for each
entity and comparing the years outside the reader. Director questions take
one extra hop. Any missing or tied year keeps the witness-bundle reader.
No cutoff is fit on this block.
"""

from pathlib import Path

from epirag.holdout_report import branch_rates
from epirag.locked_report import STRESS, cells, cost_table, fair_rows, family
from epirag.paired_tests import load

ROOT = Path(__file__).resolve().parents[1]
FAMILY_A = [
    ("epirag_exec", "debate"),
    ("epirag_exec", "centralized"),
    ("epirag_exec", "epirag_flat"),
    ("epirag_exec", "epirag_graph"),
    ("epirag_exec", "epirag_cover"),
]
METHODS = [
    "centralized",
    "debate",
    "epirag_cover",
    "epirag_flat",
    "epirag_graph",
    "epirag_exec",
]


def report(path: Path):
    rows = load(path)
    print(f"\n##### {path} rows {len(rows)}")
    grouped = fair_rows(rows)
    cells(grouped, METHODS)
    family(grouped, FAMILY_A, STRESS, "family A stress")
    branch_rates(rows, "epirag_exec")
    cost_table(grouped)


if __name__ == "__main__":
    report(ROOT / "experiments" / "results" / "records_prospect.jsonl")
