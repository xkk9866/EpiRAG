"""Print one copy of compositional and inference questions."""

import json

SEEN = set()
with open("experiments/results/records_prospect.jsonl", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if row["method"] != "epirag_cover" or row["condition"] != "asymmetry":
            continue
        if row["qtype"] not in {"compositional", "inference", "comparison", "bridge_comparison"}:
            continue
        if row["qtype"] in SEEN and row["dataset"] == "hotpotqa":
            continue
        key = (row["dataset"], row["qtype"], row["question"])
        if key in SEEN:
            continue
        SEEN.add(key)
        print(f"{row['dataset']}|{row['qtype']}|{row['question']}")
