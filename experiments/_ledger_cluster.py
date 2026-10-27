"""Question-clustered sign test on the frozen ledger family. Descriptive check."""

from collections import defaultdict
from pathlib import Path

from epirag.locked_report import STRESS, fair_rows, holm
from epirag.paired_tests import binom_two_sided, load

rows = load(Path("experiments/results/records_ledger.jsonl"))
grouped = fair_rows(rows)
family = [
    "debate",
    "centralized",
    "epirag_flat",
    "epirag_graph",
    "epirag_cover",
]
records = []
for dataset in ("hotpotqa", "2wiki"):
    for other in family:
        totals = defaultdict(int)
        for key, methods in grouped.items():
            if key[0] != dataset or key[1] not in STRESS:
                continue
            totals[key[2]] += methods["epirag_hop"]["em"] - methods[other]["em"]
        pos = sum(value > 0 for value in totals.values())
        neg = sum(value < 0 for value in totals.values())
        p = binom_two_sided(pos, pos + neg)
        records.append((dataset, other, pos, neg, p, len(totals)))
adjusted = holm([item[4] for item in records])
for item, holm_p in zip(records, adjusted):
    dataset, other, pos, neg, p, n = item
    print(f"{dataset} vs {other} questions {pos}/{neg} n={n} p={p:.4g} holm={holm_p:.4g}")
