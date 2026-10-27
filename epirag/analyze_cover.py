"""Compare claim-covering bundles with the flat filter and the baselines."""

import json
from collections import defaultdict
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("experiments/results/records_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print("rows", len(rows))
bad = [
    row
    for row in rows
    if row["answer"] in ("", "-", "td", "0", "1")
    or str(row["answer"]).lstrip().startswith("{")
    or row["f1"] is None
    or row["em"] is None
]
print("bad rows", len(bad))

buckets = defaultdict(list)
for row in rows:
    buckets[(row["dataset"], row["condition"], row["method"])].append(row)

methods = [
    "independent",
    "centralized",
    "debate",
    "collab",
    "epirag",
    "epirag_flat",
    "epirag_cover",
    "epirag_nofilter",
    "epirag_noretrieval",
    "epirag_noreliability",
]
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        print(f"\n{dataset} {condition}")
        for method in methods:
            group = buckets[(dataset, condition, method)]
            if not group:
                print(f"  {method:22} MISSING")
                continue
            f1 = 100 * sum(row["f1"] for row in group) / len(group)
            em = 100 * sum(row["em"] for row in group) / len(group)
            trig = 100 * sum(1 for row in group if row["triggered"]) / len(group)
            tok = sum(row["comm_tokens"] for row in group) / len(group)
            ece_rows = [row for row in group if row.get("confidence") is not None]
            print(
                f"  {method:22} n {len(group):3} EM {em:5.1f} F1 {f1:5.1f} "
                f"trig {trig:5.1f} tok {tok:6.1f}"
            )

index = {}
for row in rows:
    index[(row["dataset"], row["qid"], row["condition"], row["method"])] = row

print("\ncover vs flat answer changes")
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        same = gain = loss = 0
        for key, cover in index.items():
            if key[0] != dataset or key[2] != condition or key[3] != "epirag_cover":
                continue
            flat = index[(dataset, key[1], condition, "epirag_flat")]
            if cover["answer"] == flat["answer"]:
                same += 1
            elif cover["em"] > flat["em"] or (
                cover["em"] == flat["em"] and cover["f1"] > flat["f1"] + 0.02
            ):
                gain += 1
            elif cover["em"] < flat["em"] or (
                cover["em"] == flat["em"] and cover["f1"] + 0.02 < flat["f1"]
            ):
                loss += 1
            else:
                same += 1
        print(f"  {dataset:10} {condition:12} same {same:3} gain {gain:3} loss {loss:3}")

print("\ncover loses to debate or centralized while flat also loses")
shown = 0
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("asymmetry", "corruption", "incomplete", "clean", "temporal"):
        losses = []
        for key, cover in index.items():
            if key[0] != dataset or key[2] != condition or key[3] != "epirag_cover":
                continue
            debate = index[(dataset, key[1], condition, "debate")]
            cen = index[(dataset, key[1], condition, "centralized")]
            if cover["em"] == 0 and (debate["em"] == 1 or cen["em"] == 1):
                losses.append((cover, debate, cen))
        print(f"  {dataset} {condition} cover-wrong baseline-right {len(losses)}")
        for cover, debate, cen in losses[:3]:
            print("   Q", cover["question"][:110])
            print("   gold", cover["gold"], "cover", cover["answer"], "hat", cover.get("hat"))
            print("   debate", debate["answer"], "cen", cen["answer"], "locals", cover.get("local_answers"))
        shown += len(losses)
print("total such losses", shown)
