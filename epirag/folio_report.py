"""Folio block. Family fixed before this block is scored.

Family A, one reader, Holm over 12 tests. Stress-pooled exact match.
Stress is asymmetry, corruption, temporal, and incomplete.
Contrasts: epirag_derive against debate, centralized, epirag_flat,
epirag_graph, epirag_cover, and epirag_hop, on each dataset.

The same 12 contrasts are also tested by a question-clustered sign flip.
Each question contributes the sum of its four stress exact-match differences,
so the four regimes are not treated as independent items. That check was
written down before any folio row existed.

epirag_derive plans the lookups from the question alone. A chain answer is
the last span that is stated in its lookup sentences. A comparison is decided
from the two stated values, not from the raw pages. An unstated step leaves
the witness-bundle reader in place. No cutoff is fit on this block.
"""

from collections import defaultdict
from pathlib import Path

from epirag.cluster_tests import clustered_exact_p
from epirag.holdout_report import branch_rates
from epirag.locked_report import STRESS, cells, cost_table, fair_rows, family, holm
from epirag.paired_tests import load

ROOT = Path(__file__).resolve().parents[1]
FAMILY_A = [
    ("epirag_derive", "debate"),
    ("epirag_derive", "centralized"),
    ("epirag_derive", "epirag_flat"),
    ("epirag_derive", "epirag_graph"),
    ("epirag_derive", "epirag_cover"),
    ("epirag_derive", "epirag_hop"),
]
METHODS = [
    "centralized",
    "debate",
    "epirag_cover",
    "epirag_flat",
    "epirag_graph",
    "epirag_hop",
    "epirag_derive",
]


def clustered_family(grouped):
    tests = []
    for dataset in ("hotpotqa", "2wiki"):
        for left, right in FAMILY_A:
            clusters = defaultdict(int)
            for (ds, condition, qid), methods in grouped.items():
                if ds == dataset and condition in STRESS and left in methods and right in methods:
                    clusters[qid] += int(methods[left]["em"] - methods[right]["em"])
            p_value, nonzero = clustered_exact_p(list(clusters.values()))
            pos = sum(value > 0 for value in clusters.values())
            neg = sum(value < 0 for value in clusters.values())
            tests.append((dataset, right, pos, neg, nonzero, p_value))
    print("\n-- question-clustered sign flip")
    for item, corrected in zip(tests, holm([item[-1] for item in tests])):
        dataset, right, pos, neg, nonzero, p_value = item
        print(
            f"{dataset:9} epirag_derive vs {right:16} "
            f"questions {pos}/{neg} nonzero {nonzero} p={p_value:.4g} holm={corrected:.4g}"
        )


def report(path: Path):
    rows = load(path)
    print(f"\n##### {path} rows {len(rows)}")
    grouped = fair_rows(rows)
    cells(grouped, METHODS)
    family(grouped, FAMILY_A, STRESS, "family A stress")
    clustered_family(grouped)
    branch_rates(rows, "epirag_derive")
    cost_table(grouped)


if __name__ == "__main__":
    report(ROOT / "experiments" / "results" / "records_folio.jsonl")
