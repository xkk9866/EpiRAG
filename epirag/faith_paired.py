"""McNemar on FaithEval conflict decisions."""

import json
import math
from collections import defaultdict
from pathlib import Path


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2**n))


rows = []
with Path("experiments/results/records_faith.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            rows.append(json.loads(line))
by_q = defaultdict(dict)
for row in rows:
    by_q[row["qid"]][row["method"]] = row
print("n", len(by_q))
pairs = [
    ("epirag_cover", "epirag_flat"),
    ("epirag_cover", "centralized"),
    ("epirag_flat", "centralized"),
    ("epirag_cover", "debate"),
    ("debate", "centralized"),
]
for left, right in pairs:
    both = [item for item in by_q.values() if left in item and right in item]
    a_only = sum(1 for item in both if item[left]["conflict"] > item[right]["conflict"])
    b_only = sum(1 for item in both if item[right]["conflict"] > item[left]["conflict"])
    rate_a = 100 * sum(item[left]["conflict"] for item in both) / len(both)
    rate_b = 100 * sum(item[right]["conflict"] for item in both) / len(both)
    print(
        f"{left:16} vs {right:16} conflict {rate_a:.1f}/{rate_b:.1f} "
        f"only {a_only}/{b_only} p={binom_two_sided(a_only, a_only + b_only):.4f}"
    )
