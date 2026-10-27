"""F1, trigger rate, and where typed belief revision loses to the flat filter."""

import json
from collections import defaultdict
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("experiments/results/records_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print("rows", len(rows), "json", sum(1 for row in rows if str(row["answer"]).lstrip().startswith("{")))

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
    "epirag_nofilter",
    "epirag_noretrieval",
    "epirag_noreliability",
]
print("\nF1 and trigger")
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        print(f"\n{dataset} {condition}")
        for method in methods:
            group = buckets[(dataset, condition, method)]
            f1 = 100 * sum(row["f1"] for row in group) / len(group)
            em = 100 * sum(row["em"] for row in group) / len(group)
            trig = 100 * sum(1 for row in group if row["triggered"]) / len(group)
            tok = sum(row["comm_tokens"] for row in group) / len(group)
            print(f"  {method:22} EM {em:5.1f} F1 {f1:5.1f} trig {trig:5.1f} tok {tok:6.1f}")

index = {}
for row in rows:
    index[(row["dataset"], row["qid"], row["condition"], row["method"])] = row

print("\nflat correct, typed wrong, hotpot asymmetry")
shown = 0
for (dataset, qid, condition, method), typed in index.items():
    if method != "epirag" or dataset != "hotpotqa" or condition != "asymmetry":
        continue
    flat = index[(dataset, qid, condition, "epirag_flat")]
    if flat["em"] == 1 and typed["em"] == 0:
        print("Q", typed["question"][:120])
        print(" gold", typed["gold"])
        print(" typed", typed["answer"], "hat", typed.get("hat"), "trig", typed["triggered"], "claims", typed.get("claims"))
        print(" flat ", flat["answer"], "hat", flat.get("hat"), "trig", flat["triggered"], "claims", flat.get("claims"))
        shown += 1
        if shown >= 8:
            break
print("count shown", shown)
