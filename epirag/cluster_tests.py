"""Exact question-clustered randomization tests for the frozen final block.

Each question appears in four stress regimes. A single sign flip is applied to
the sum of its four paired exact-match differences, preserving dependence
between regimes. This is post-processing of existing records, not a new run.
"""

from collections import Counter, defaultdict
from pathlib import Path

from epirag.locked_report import STRESS, fair_rows, holm
from epirag.paired_tests import load


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "experiments" / "results" / "records_final.jsonl"
DATASETS = ("hotpotqa", "2wiki")
BASELINES = ("epirag_flat", "debate", "centralized", "epirag_graph")


def clustered_exact_p(differences):
    """Two-sided sign-flip p-value for integer cluster totals."""
    weights = [abs(int(value)) for value in differences if value]
    observed = abs(sum(int(value) for value in differences))
    counts = Counter({0: 1})
    for weight in weights:
        next_counts = Counter()
        for total, count in counts.items():
            next_counts[total + weight] += count
            next_counts[total - weight] += count
        counts = next_counts
    extreme = sum(count for total, count in counts.items() if abs(total) >= observed)
    return extreme / (2 ** len(weights)), len(weights)


def family(grouped, candidate):
    tests = []
    for dataset in DATASETS:
        for baseline in BASELINES:
            clusters = defaultdict(int)
            for (ds, condition, qid), methods in grouped.items():
                if ds == dataset and condition in STRESS:
                    clusters[qid] += int(methods[candidate]["em"] - methods[baseline]["em"])
            p, nonzero = clustered_exact_p(list(clusters.values()))
            tests.append({
                "dataset": dataset, "candidate": candidate, "baseline": baseline,
                "question_clusters": len(clusters), "nonzero_clusters": nonzero,
                "net_correct": sum(clusters.values()), "p": p,
            })
    for test, corrected in zip(tests, holm([test["p"] for test in tests])):
        test["holm_p"] = corrected
    return tests


def main():
    grouped = fair_rows(load(ROWS))
    for name, candidate in (("Pre-specified commitment", "epirag_bridge"),
                            ("Exploratory witness", "epirag_cover")):
        print(name)
        for row in family(grouped, candidate):
            print(f"{row['dataset']:8} vs {row['baseline']:15} "
                  f"clusters={row['question_clusters']:3} "
                  f"nonzero={row['nonzero_clusters']:3} "
                  f"net={row['net_correct']:4} "
                  f"p={row['p']:.6g} Holm={row['holm_p']:.6g}")


if __name__ == "__main__":
    main()
