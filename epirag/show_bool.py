"""Show the questions changed by boolean canonicalization."""

import json
from pathlib import Path

from epirag.metrics import exact_match, normalize_answer

BOOL = {"yes": "yes", "no": "no", "true": "yes", "false": "no"}


def canon_bool(text: str):
    return BOOL.get(normalize_answer(text))


def rule_bool(row):
    answer = row["answer"]
    mapped = canon_bool(answer)
    if mapped is None:
        return answer
    claims = row.get("claims") or []
    claim_bools = {canon_bool(c) for c in claims}
    claim_bools.discard(None)
    if mapped in claim_bools:
        for claim in claims:
            if canon_bool(claim) == mapped and normalize_answer(claim) in {"yes", "no"}:
                return claim
        return mapped
    return answer


rows = [
    json.loads(line)
    for line in Path("experiments/results/records_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip() and json.loads(line)["method"] == "epirag_cover"
]
for row in rows:
    pred = rule_bool(row)
    if normalize_answer(pred) == normalize_answer(row["answer"]):
        continue
    print(row["dataset"], row["condition"], "EM", row["em"], "->", exact_match(pred, row["gold"]))
    print(" Q", row["question"][:140])
    print(" gold", row["gold"], "| reader", row["answer"], "| pred", pred, "| claims", row.get("claims"))
