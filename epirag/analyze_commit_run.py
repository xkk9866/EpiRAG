"""Separate polar binding from support-gate overrides."""

import json
from pathlib import Path

from epirag.metrics import exact_match, normalize_answer, token_f1

rows = [
    json.loads(line)
    for line in Path("experiments/results/records_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
by = {}
for row in rows:
    by[(row["dataset"], row["qid"], row["condition"], row["method"])] = row

methods = [
    "epirag_commit",
    "epirag_commit_noret",
    "epirag_commit_norel",
    "epirag_cover",
    "centralized",
    "debate",
]
print("F1")
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        print(f"\n{dataset} {condition}")
        for method in methods:
            group = [
                row
                for (ds, _q, cond, meth), row in by.items()
                if ds == dataset and cond == condition and meth == method
            ]
            em = 100 * sum(row["em"] for row in group) / len(group)
            f1 = 100 * sum(row["f1"] for row in group) / len(group)
            trig = 100 * sum(1 for row in group if row["triggered"]) / len(group)
            tok = sum(row["comm_tokens"] for row in group) / len(group)
            print(f"  {method:22} EM {em:5.1f} F1 {f1:5.1f} trig {trig:5.1f} tok {tok:6.1f}")

print("\ngate changes vs cover reader")
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        gain = loss = same = 0
        for (ds, qid, cond, meth), row in by.items():
            if ds != dataset or cond != condition or meth != "epirag_commit":
                continue
            cover = by[(ds, qid, cond, "epirag_cover")]
            if normalize_answer(row["answer"]) == normalize_answer(cover["answer"]):
                same += 1
                continue
            if row["em"] > cover["em"] or (
                row["em"] == cover["em"] and row["f1"] > cover["f1"] + 0.02
            ):
                gain += 1
            elif row["em"] < cover["em"] or (
                row["em"] == cover["em"] and row["f1"] + 0.02 < cover["f1"]
            ):
                loss += 1
            else:
                same += 1
        print(f"  {dataset:10} {condition:12} same {same:3} gain {gain:3} loss {loss:3}")

print("\n2wiki incomplete losses")
shown = 0
for (ds, qid, cond, meth), row in by.items():
    if not (ds == "2wiki" and cond == "incomplete" and meth == "epirag_commit"):
        continue
    cover = by[(ds, qid, cond, "epirag_cover")]
    if row["em"] >= cover["em"]:
        continue
    print(" gold", row["gold"])
    print(" commit", row["answer"], "| reader", row.get("reader_answer"), "| locals", row.get("local_answers"))
    print(" cover", cover["answer"], "Q", row["question"][:120])
    shown += 1
    if shown >= 12:
        break
print("shown", shown)
