"""Post-hoc decision rules on stored cover records. No gold is used by the rules."""

import json
from collections import defaultdict
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

BOOL = {
    "yes": "yes",
    "no": "no",
    "true": "yes",
    "false": "no",
}


def canon_bool(text: str) -> str | None:
    key = normalize_answer(text)
    return BOOL.get(key)


def unanimous(row) -> str | None:
    locals_ = row.get("local_answers") or []
    keys = [normalize_answer(a) for a in locals_ if normalize_answer(a)]
    if len(keys) >= 2 and len(set(keys)) == 1:
        return locals_[0]
    return None


def rule_lock(row):
    locked = unanimous(row)
    if locked is None:
        return row["answer"]
    if token_f1(row["answer"], locked) >= 0.5:
        return row["answer"]
    return locked


def rule_bool(row):
    answer = row["answer"]
    mapped = canon_bool(answer)
    if mapped is None:
        return answer
    claims = row.get("claims") or []
    claim_bools = {canon_bool(c) for c in claims}
    claim_bools.discard(None)
    if mapped in claim_bools:
        # Prefer the agents' surface form when they already said yes/no.
        for claim in claims:
            if canon_bool(claim) == mapped and normalize_answer(claim) in {"yes", "no"}:
                return claim
        return mapped
    return answer


def rule_both(row):
    return rule_bool({"answer": rule_lock(row), "claims": row.get("claims"), "local_answers": row.get("local_answers")})


def score(name, fn, method="epirag_cover"):
    print(f"\n{name}")
    for dataset in ("hotpotqa", "2wiki"):
        for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
            group = [
                row
                for (ds, _qid, cond, meth), row in by.items()
                if ds == dataset and cond == condition and meth == method
            ]
            pred = [fn(row) for row in group]
            em = 100 * sum(exact_match(p, row["gold"]) for p, row in zip(pred, group)) / len(group)
            f1 = 100 * sum(token_f1(p, row["gold"]) for p, row in zip(pred, group)) / len(group)
            base_em = 100 * sum(row["em"] for row in group) / len(group)
            base_f1 = 100 * sum(row["f1"] for row in group) / len(group)
            changed = sum(normalize_answer(p) != normalize_answer(row["answer"]) for p, row in zip(pred, group))
            print(
                f"  {dataset:10} {condition:12} EM {em:5.1f} ({base_em:5.1f}) "
                f"F1 {f1:5.1f} ({base_f1:5.1f}) changed {changed}"
            )


score("lock unanimous when reader disagrees", rule_lock)
score("boolean surface from claims", rule_bool)
score("lock then boolean", rule_both)

# Best baseline in each cell for reference
print("\nbest baseline EM/F1")
for dataset in ("hotpotqa", "2wiki"):
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        best = None
        for method in ("independent", "centralized", "debate", "collab", "epirag_flat", "epirag_cover"):
            group = [
                row
                for (ds, _qid, cond, meth), row in by.items()
                if ds == dataset and cond == condition and meth == method
            ]
            em = 100 * sum(row["em"] for row in group) / len(group)
            f1 = 100 * sum(row["f1"] for row in group) / len(group)
            if best is None or f1 > best[0]:
                best = (f1, em, method)
        print(f"  {dataset:10} {condition:12} {best[2]:16} EM {best[1]:5.1f} F1 {best[0]:5.1f}")
