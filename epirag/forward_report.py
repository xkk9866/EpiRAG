"""Forward block. Family fixed before this block is scored.

Family A, one reader, Holm over 10 tests. Stress-pooled exact match.
Stress is asymmetry, corruption, temporal, and incomplete.
Contrasts: epirag_chain against debate, centralized, epirag_flat,
epirag_graph, and epirag_cover, on each dataset.

epirag_chain reads the witness pages once and copies a bridge entity of at
most eight tokens, the frozen claim-length cap. A second read answers from
the six-page pool conditioned on that entity. No cutoff is fit on this block.
"""

from pathlib import Path

from epirag.holdout_report import branch_rates
from epirag.locked_report import STRESS, cells, cost_table, fair_rows, family
from epirag.paired_tests import load

ROOT = Path(__file__).resolve().parents[1]
FAMILY_A = [
    ("epirag_chain", "debate"),
    ("epirag_chain", "centralized"),
    ("epirag_chain", "epirag_flat"),
    ("epirag_chain", "epirag_graph"),
    ("epirag_chain", "epirag_cover"),
]
METHODS = [
    "centralized",
    "debate",
    "epirag_cover",
    "epirag_flat",
    "epirag_graph",
    "epirag_chain",
]


def report(path: Path):
    rows = load(path)
    print(f"\n##### {path} rows {len(rows)}")
    grouped = fair_rows(rows)
    cells(grouped, METHODS)
    family(grouped, FAMILY_A, STRESS, "family A stress")
    branch_rates(rows, "epirag_chain")
    cost_table(grouped)


if __name__ == "__main__":
    report(ROOT / "experiments" / "results" / "records_forward.jsonl")
