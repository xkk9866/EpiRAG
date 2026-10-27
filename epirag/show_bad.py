"""Show stored answers that failed JSON parsing."""

import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("experiments/results/records.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
shown = 0
for row in rows:
    text = (row["answer"] or "").strip()
    if not text.startswith("{"):
        continue
    print("---", row["dataset"], row["method"], row["condition"], "gold:", row["gold"])
    print(text)
    print()
    shown += 1
    if shown >= 12:
        break
