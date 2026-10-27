import json

rows = [json.loads(line) for line in open("experiments/results/smoke.jsonl", encoding="utf-8")]
print("n", len(rows))
print("Q", rows[0]["question"])
print("gold", rows[0]["gold"])
for row in rows:
    if row["dataset"] != "hotpotqa":
        continue
    if row["condition"] not in {"clean", "corruption", "incomplete", "asymmetry"}:
        continue
    if row["method"] not in {"independent", "centralized", "debate", "collab", "epirag", "vote"}:
        continue
    print(
        f"{row['condition']:12} {row['method']:14} em={row['em']:.0f} f1={row['f1']:.2f} "
        f"conf={row['confidence']:.2f} trig={row['triggered']} msg={row['messages']} "
        f"tok={row['comm_tokens']}"
    )
    print("   ans:", row["answer"][:160])
print("--- 2wiki gold", [r for r in rows if r["dataset"]=="2wiki"][0]["gold"])
for row in rows:
    if row["dataset"] != "2wiki" or row["condition"] != "clean":
        continue
    if row["method"] in {"centralized", "epirag", "independent"}:
        print(row["method"], row["answer"][:160], "f1", round(row["f1"], 3))
