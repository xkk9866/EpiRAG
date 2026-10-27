import json
from collections import defaultdict

path = "experiments/results/records.jsonl"
rows = []
try:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
except FileNotFoundError:
    print("no records yet")
    raise SystemExit
print("rows", len(rows))
main = {"independent", "centralized", "debate", "collab", "epirag"}
buckets = defaultdict(list)
for row in rows:
    if row["method"] in main:
        buckets[(row["dataset"], row["condition"], row["method"])].append(row)
# complete questions: qids with all 5 conditions for epirag
by_q = defaultdict(set)
for row in rows:
    if row["method"] == "epirag":
        by_q[(row["dataset"], row["qid"])].add(row["condition"])
complete = {k for k, v in by_q.items() if len(v) == 5}
print("complete questions", len(complete))
for dataset in ("hotpotqa", "2wiki"):
    print("==", dataset)
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        parts = []
        for method in ("independent", "centralized", "debate", "collab", "epirag"):
            items = [
                r
                for r in buckets[(dataset, condition, method)]
                if (r["dataset"], r["qid"]) in complete
            ]
            if not items:
                continue
            em = sum(r["em"] for r in items) / len(items)
            f1 = sum(r["f1"] for r in items) / len(items)
            parts.append(f"{method[:4]} EM {em:.3f} F1 {f1:.3f}")
        if parts:
            print(condition, "n", len(items), " | ".join(parts))
