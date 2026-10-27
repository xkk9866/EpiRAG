"""Where the coordinated claim is right and the reader still flips it."""

import json
import re
import string
from collections import defaultdict
from pathlib import Path

from epirag.metrics import exact_match, token_f1


def norm(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


rows = [
    json.loads(line)
    for line in Path("experiments/results/records_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip() and json.loads(line)["method"] == "epirag_cover"
]
print("cover rows", len(rows))

for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        group = [r for r in rows if r["dataset"] == dataset and r["condition"] == condition]
        hat_em = 100 * sum(exact_match(r.get("hat") or "", r["gold"]) for r in group) / len(group)
        ans_em = 100 * sum(r["em"] for r in group) / len(group)
        hat_f1 = 100 * sum(token_f1(r.get("hat") or "", r["gold"]) for r in group) / len(group)
        ans_f1 = 100 * sum(r["f1"] for r in group) / len(group)
        both = sum(1 for r in group if r["em"] == 1 and exact_match(r.get("hat") or "", r["gold"]))
        hat_only = sum(1 for r in group if r["em"] == 0 and exact_match(r.get("hat") or "", r["gold"]))
        ans_only = sum(1 for r in group if r["em"] == 1 and not exact_match(r.get("hat") or "", r["gold"]))
        print(
            f"{dataset:10} {condition:12} hatEM {hat_em:5.1f} ansEM {ans_em:5.1f} "
            f"hatF1 {hat_f1:5.1f} ansF1 {ans_f1:5.1f} "
            f"both {both:3} hat_only {hat_only:3} ans_only {ans_only:3}"
        )

print("\nlocal majority vs hat vs answer")
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        group = [r for r in rows if r["dataset"] == dataset and r["condition"] == condition]
        maj_right = hat_right_reader_wrong = 0
        for row in group:
            counts = defaultdict(int)
            for ans in row.get("local_answers") or []:
                counts[norm(ans)] += 1
            if not counts:
                continue
            top = max(counts, key=counts.get)
            # recover an original string with that key
            original = next(a for a in row["local_answers"] if norm(a) == top)
            if exact_match(original, row["gold"]):
                maj_right += 1
            if exact_match(row.get("hat") or "", row["gold"]) and row["em"] == 0:
                hat_right_reader_wrong += 1
        print(
            f"{dataset:10} {condition:12} majority_EM_count {maj_right:3} "
            f"hat_right_reader_wrong {hat_right_reader_wrong:3} n {len(group)}"
        )

print("\nmargin when hat right and reader wrong")
bins = defaultdict(lambda: [0, 0])
for row in rows:
    margin = row.get("margin")
    if margin is None:
        continue
    bucket = f"{min(int(margin * 10) / 10, 0.9):.1f}"
    hat_ok = exact_match(row.get("hat") or "", row["gold"])
    if hat_ok and row["em"] == 0:
        bins[bucket][0] += 1
    elif (not hat_ok) and row["em"] == 1:
        bins[bucket][1] += 1
for bucket in sorted(bins):
    print(f"  margin {bucket} hat_only {bins[bucket][0]:3} reader_only {bins[bucket][1]:3}")
