"""Descriptive splits of the frozen ledger block. No cutoff is changed."""

from collections import defaultdict
from pathlib import Path

from epirag.locked_report import STRESS, fair_rows
from epirag.paired_tests import load

rows = load(Path("experiments/results/records_ledger.jsonl"))
grouped = fair_rows(rows)

print("COLLAB CELLS")
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete", "collusion"):
        items = [
            methods["collab"]
            for key, methods in grouped.items()
            if key[0] == dataset and key[1] == condition and "collab" in methods
        ]
        em = 100 * sum(item["em"] for item in items) / len(items)
        f1 = 100 * sum(item["f1"] for item in items) / len(items)
        tok = sum(item["tokens"] for item in items) / len(items)
        print(f"{dataset} {condition} n={len(items)} {em:.1f}/{f1:.1f} ctok {tok:.0f}")

print("BY DECISION STRESS")
for dataset in ("hotpotqa", "2wiki"):
    buckets = defaultdict(list)
    for key, methods in grouped.items():
        if key[0] != dataset or key[1] not in STRESS:
            continue
        if "epirag_hop" not in methods or "epirag_cover" not in methods:
            continue
        decision = methods["epirag_hop"]["decision"] or "none"
        buckets[decision].append((methods["epirag_hop"], methods["epirag_cover"]))
    for decision, pairs in sorted(buckets.items()):
        n = len(pairs)
        hop_em = 100 * sum(left["em"] for left, _right in pairs) / n
        cover_em = 100 * sum(right["em"] for _left, right in pairs) / n
        hop_f1 = 100 * sum(left["f1"] for left, _right in pairs) / n
        cover_f1 = 100 * sum(right["f1"] for _left, right in pairs) / n
        print(
            f"{dataset} {decision} n={n} hop {hop_em:.1f}/{hop_f1:.1f} cover {cover_em:.1f}/{cover_f1:.1f}"
        )

print("BY TYPE STRESS EM")
types = {}
for row in rows:
    if row["method"] == "epirag_hop":
        types[(row["dataset"], row["qid"])] = row["qtype"]
for dataset in ("hotpotqa", "2wiki"):
    buckets = defaultdict(lambda: {"hop": [], "cover": []})
    for key, methods in grouped.items():
        if key[0] != dataset or key[1] not in STRESS:
            continue
        qtype = types[(key[0], key[2])]
        buckets[qtype]["hop"].append(methods["epirag_hop"]["em"])
        buckets[qtype]["cover"].append(methods["epirag_cover"]["em"])
    for qtype, vals in sorted(buckets.items()):
        n = len(vals["hop"])
        hop = 100 * sum(vals["hop"]) / n
        cover = 100 * sum(vals["cover"]) / n
        print(f"{dataset} {qtype} n={n} hop {hop:.1f} cover {cover:.1f}")
