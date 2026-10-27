"""Selective commitment: keep the reader unless it switches to a low-belief claim."""

import json
from pathlib import Path

from epirag.metrics import exact_match, normalize_answer, token_f1

rows = [
    json.loads(line)
    for line in Path("experiments/results/records_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip() and json.loads(line)["method"] == "epirag_cover"
]

BOOL = {"yes": "yes", "no": "no", "true": "yes", "false": "no"}


def canon(text: str) -> str:
    key = normalize_answer(text)
    return BOOL.get(key, key)


def match_claim(answer: str, claims: list[str]) -> str | None:
    target = canon(answer)
    if not target:
        return None
    for claim in claims:
        if canon(claim) == target or token_f1(answer, claim) >= 0.8:
            return claim
    return None


def decide(row, margin_lock: float) -> str:
    answer = row["answer"]
    claims = row.get("claims") or []
    hat = row.get("hat") or answer
    margin = float(row.get("margin") or 0.0)
    # Polar questions keep the yes/no lexicon once any claim used it.
    if canon(answer) in {"yes", "no"}:
        for claim in claims:
            if canon(claim) == canon(answer) and normalize_answer(claim) in {"yes", "no"}:
                answer = claim
                break
        else:
            if canon(answer) in {canon(c) for c in claims}:
                answer = canon(answer)
    matched = match_claim(answer, claims)
    if matched is None:
        return answer
    if canon(matched) == canon(hat):
        return matched if normalize_answer(matched) in {"yes", "no"} else answer
    if margin >= margin_lock:
        return hat
    return answer


def report(margin_lock: float) -> None:
    print(f"\nlock margin {margin_lock}")
    for dataset in ("hotpotqa", "2wiki"):
        for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
            group = [r for r in rows if r["dataset"] == dataset and r["condition"] == condition]
            pred = [decide(r, margin_lock) for r in group]
            em = 100 * sum(exact_match(p, r["gold"]) for p, r in zip(pred, group)) / len(group)
            f1 = 100 * sum(token_f1(p, r["gold"]) for p, r in zip(pred, group)) / len(group)
            base_f1 = 100 * sum(r["f1"] for r in group) / len(group)
            print(f"  {dataset:10} {condition:12} EM {em:5.1f} F1 {f1:5.1f} baseF1 {base_f1:5.1f}")


for lock in (0.35, 0.55, 0.75, 1.01):
    report(lock)
