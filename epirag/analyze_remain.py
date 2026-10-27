"""Residual errors after yes/no canonicalization."""

import json
from collections import Counter
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

BOOL = {"yes": "yes", "no": "no", "true": "yes", "false": "no"}


def canon(text: str) -> str:
    key = normalize_answer(text)
    return BOOL.get(key, key)


def bound(row) -> str:
    answer = row["answer"]
    if canon(answer) in {"yes", "no"}:
        for claim in row.get("claims") or []:
            if canon(claim) == canon(answer) and normalize_answer(claim) in {"yes", "no"}:
                return claim
        return canon(answer)
    return answer


def locals_right(row) -> bool:
    vals = [a for a in row.get("local_answers") or [] if normalize_answer(a)]
    if not vals:
        return False
    return all(exact_match(a, row["gold"]) for a in vals)


kinds = Counter()
examples = {k: [] for k in ("unanimous_flip", "near_miss", "pool_has_it", "nobody")}
for key, row in by.items():
    if key[3] != "epirag_cover":
        continue
    pred = bound(row)
    if exact_match(pred, row["gold"]):
        continue
    debate = by[(key[0], key[1], key[2], "debate")]
    cen = by[(key[0], key[1], key[2], "centralized")]
    f1 = token_f1(pred, row["gold"])
    if locals_right(row):
        kind = "unanimous_flip"
    elif f1 >= 0.5:
        kind = "near_miss"
    elif cen["em"] == 1 or debate["em"] == 1:
        kind = "pool_has_it"
    else:
        kind = "nobody"
    kinds[kind] += 1
    if len(examples[kind]) < 6:
        examples[kind].append((row, pred, debate["answer"], cen["answer"], f1))

print(dict(kinds))
for kind, items in examples.items():
    print(f"\n== {kind}")
    for row, pred, debate, cen, f1 in items:
        print(row["dataset"], row["condition"], f"f1 {f1:.2f}", "trig", row["triggered"])
        print(" Q", row["question"][:130])
        print(" gold", row["gold"])
        print(" pred", pred, "| hat", row.get("hat"), "| locals", row.get("local_answers"))
        print(" debate", debate, "| cen", cen)
