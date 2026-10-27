"""Count answers that were stored as truncated JSON rather than a span."""

import json
from collections import defaultdict
from pathlib import Path


def bad(answer: str) -> bool:
    text = (answer or "").strip()
    return text.startswith("{") or '"answer"' in text[:80]


def main():
    rows = [
        json.loads(line)
        for line in Path("experiments/results/records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["condition"])].append(row)
    print(f"total {len(rows)} bad {sum(1 for row in rows if bad(row['answer']))}")
    for key, group in sorted(grouped.items()):
        broken = [row for row in group if bad(row["answer"])]
        if not broken:
            continue
        em_zero = sum(1 for row in broken if row["em"] == 0)
        print(
            f"{key[0]:10} {key[1]:24} {key[2]:12} "
            f"n={len(group):3} bad={len(broken):3} em0={em_zero}"
        )
    local_n = 0
    local_bad = 0
    for row in rows:
        if row["method"] != "epirag":
            continue
        for answer in row.get("local_answers") or []:
            local_n += 1
            if bad(answer):
                local_bad += 1
    print(f"local bad {local_bad} / {local_n}")


if __name__ == "__main__":
    main()
