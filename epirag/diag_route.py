"""Simulate routing the real pool and the real witness bundle.

Pool answers are the 6-page centralized reader. Witness answers are the
reliability-on cover reader. The gate record only supplies the frozen
contradiction decision and the belief margin. This is a development
estimate on the holdout, not a reported test: a later run must actually
read the selected pages.
"""

import json
from collections import defaultdict

from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
rows = defaultdict(dict)
with open("experiments/results/records_holdout.jsonl", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if row["method"] not in ("centralized", "epirag_cover", "epirag_gate"):
            continue
        rows[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row


def canon(row):
    if row["answer"] == "conflict":
        return row["answer"]
    return canonicalize_polar(row["question"], row["answer"])


def evaluate(margin_cut):
    print("margin_cut", margin_cut)
    for dataset in ("hotpotqa", "2wiki"):
        for scope, pred in (
            ("stress", lambda c: c in STRESS),
            ("all", lambda c: True),
        ):
            n = hit = pool_hit = cover_hit = routed_cover = 0
            for key, methods in rows.items():
                if key[0] != dataset or not pred(key[2]):
                    continue
                pool = methods["centralized"]
                cover = methods["epirag_cover"]
                gate = methods["epirag_gate"]
                use_cover = gate["decision"] == "witness" and float(gate["margin"] or 0.0) >= margin_cut
                chosen = canon(cover if use_cover else pool)
                n += 1
                hit += exact_match(chosen, pool["gold"])
                pool_hit += exact_match(canon(pool), pool["gold"])
                cover_hit += exact_match(canon(cover), cover["gold"])
                routed_cover += int(use_cover)
            print(
                dataset,
                scope,
                "n", n,
                "route", round(100 * hit / n, 2),
                "pool", round(100 * pool_hit / n, 2),
                "cover", round(100 * cover_hit / n, 2),
                "cover_rate", round(100 * routed_cover / n, 1),
            )


for cut in (0.0, 0.10, 0.20, 0.30, 0.45):
    evaluate(cut)
