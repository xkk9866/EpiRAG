"""Print cover-vs-pool disagreements on the forward stress block."""

import json
from collections import defaultdict

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
KEEP = {"epirag_cover", "centralized", "epirag_flat", "epirag_graph", "debate"}


def main():
    grouped = defaultdict(dict)
    with open("experiments/results/records_forward.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS or row["method"] not in KEEP:
                continue
            grouped[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
    shown = defaultdict(int)
    for key, methods in grouped.items():
        if "epirag_cover" not in methods or "centralized" not in methods:
            continue
        cover = methods["epirag_cover"]
        pool = methods["centralized"]
        if int(cover["em"]) == int(pool["em"]):
            continue
        dataset = key[0]
        kind = "cover_only" if cover["em"] > pool["em"] else "pool_only"
        bucket = f"{dataset}:{kind}"
        if shown[bucket] >= 8:
            continue
        shown[bucket] += 1
        flat = methods.get("epirag_flat", {})
        graph = methods.get("epirag_graph", {})
        print("=" * 20, bucket, cover["condition"], cover["qtype"])
        print("Q:", cover["question"][:180])
        print("gold:", cover["gold"])
        print("cover:", cover["answer"], "| pool:", pool["answer"])
        print("flat:", flat.get("answer"), "| graph:", graph.get("answer"))
        print("locals:", cover.get("local_answers"))
        print("claims:", cover.get("claims"))


if __name__ == "__main__":
    main()
