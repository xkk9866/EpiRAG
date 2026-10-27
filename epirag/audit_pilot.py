"""Compare typed EpiRAG with the flat filter on the pilot slice."""

import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("experiments/results/pilot_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
index = {}
for row in rows:
    index[(row["dataset"], row["qid"], row["condition"], row["method"])] = row

bad_json = sum(1 for row in rows if (row["answer"] or "").strip().startswith("{"))
print("rows", len(rows), "json_answers", bad_json)

print("\nflat better than typed")
for key, typed in index.items():
    if key[3] != "epirag":
        continue
    flat = index.get((key[0], key[1], key[2], "epirag_flat"))
    if not flat or flat["f1"] <= typed["f1"] + 0.05:
        continue
    print(
        f"{key[0]} {key[2]} gap={flat['f1']-typed['f1']:.2f} "
        f"trig {typed.get('triggered')}/{flat.get('triggered')} "
        f"gapv {typed.get('coverage_gap')}"
    )
    print(" Q", typed["question"][:110])
    print(" gold", typed["gold"])
    print(" typed", typed["answer"], "claims", typed.get("claims"), "hat", typed.get("hat"))
    print(" flat ", flat["answer"], "claims", flat.get("claims"), "hat", flat.get("hat"))
    print()
