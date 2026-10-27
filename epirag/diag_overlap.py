"""Diagnose where the pool and the witness reader disagree. Not a reported test."""

import json
from collections import defaultdict

from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, token_f1

STRESS = ("asymmetry", "corruption", "temporal", "incomplete")
rows = defaultdict(dict)
with open("experiments/results/records_holdout.jsonl", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if row["condition"] not in STRESS:
            continue
        if row["method"] not in ("centralized", "epirag_cover", "debate", "epirag_gate"):
            continue
        rows[(row["dataset"], row["condition"], row["qid"])][row["method"]] = row


def canon(row):
    answer = row["answer"]
    if answer == "conflict":
        return answer
    return canonicalize_polar(row["question"], answer)


def best_local_f1(answer, locals_):
    if not locals_:
        return 0.0
    return max(token_f1(answer, loc) for loc in locals_)


def n_local(answer, locals_):
    return sum(token_f1(answer, loc) >= 0.5 for loc in locals_)


def rule_pool_default(pa, ca, locals_):
    """Keep the pool span unless only the witness span matches a local proposal."""
    if token_f1(pa, ca) >= 0.999:
        return pa
    p_hit = n_local(pa, locals_)
    c_hit = n_local(ca, locals_)
    if c_hit > 0 and p_hit == 0:
        return ca
    return pa


def rule_max_local(pa, ca, locals_):
    """Keep the span with strictly stronger overlap to a local proposal; ties stay with the pool."""
    if token_f1(pa, ca) >= 0.999:
        return pa
    if best_local_f1(ca, locals_) > best_local_f1(pa, locals_):
        return ca
    return pa


for dataset in ("hotpotqa", "2wiki"):
    n = same = near = far = 0
    near_pool = near_cover = near_both = near_neither = 0
    far_pool = far_cover = far_both = far_neither = 0
    far_pool_local = far_cover_local = far_both_local = 0
    for key, methods in rows.items():
        if key[0] != dataset:
            continue
        pool = methods["centralized"]
        cover = methods["epirag_cover"]
        n += 1
        pa, ca = canon(pool), canon(cover)
        if exact_match(pa, pool["gold"]) and exact_match(ca, cover["gold"]):
            pass
        f1 = token_f1(pa, ca)
        pok = exact_match(pa, pool["gold"])
        cok = exact_match(ca, cover["gold"])
        if f1 >= 0.999 or pa == ca:
            same += 1
            continue
        locals_ = [canonicalize_polar(pool["question"], a) for a in pool["local_answers"]]
        if f1 >= 0.5:
            near += 1
            near_pool += pok and not cok
            near_cover += cok and not pok
            near_both += pok and cok
            near_neither += not pok and not cok
        else:
            far += 1
            far_pool += pok and not cok
            far_cover += cok and not pok
            far_both += pok and cok
            far_neither += not pok and not cok
            p_local = any(token_f1(pa, loc) >= 0.5 for loc in locals_ if loc)
            c_local = any(token_f1(ca, loc) >= 0.5 for loc in locals_ if loc)
            if pok and not cok and p_local and not c_local:
                far_pool_local += 1
            if cok and not pok and c_local and not p_local:
                far_cover_local += 1
            if (pok or cok) and p_local and c_local:
                far_both_local += 1
    print(
        dataset,
        "n", n,
        "same", same,
        "near", near,
        "near_only_pool", near_pool,
        "near_only_cover", near_cover,
        "near_both", near_both,
        "near_neither", near_neither,
        "far", far,
        "far_only_pool", far_pool,
        "far_only_cover", far_cover,
        "far_both", far_both,
        "far_neither", far_neither,
        "far_pool_has_local", far_pool_local,
        "far_cover_has_local", far_cover_local,
        "far_both_have_local", far_both_local,
    )


def score_rule(name, fn):
    print("RULE", name)
    for dataset in ("hotpotqa", "2wiki"):
        n = hit = pool_hit = cover_hit = 0
        for key, methods in rows.items():
            if key[0] != dataset:
                continue
            pool = methods["centralized"]
            cover = methods["epirag_cover"]
            pa, ca = canon(pool), canon(cover)
            locals_ = [
                canonicalize_polar(pool["question"], answer)
                for answer in pool["local_answers"]
                if answer
            ]
            chosen = fn(pa, ca, locals_)
            n += 1
            hit += exact_match(chosen, pool["gold"])
            pool_hit += exact_match(pa, pool["gold"])
            cover_hit += exact_match(ca, cover["gold"])
        print(
            dataset,
            "n", n,
            "rule", round(100 * hit / n, 2),
            "pool", round(100 * pool_hit / n, 2),
            "cover", round(100 * cover_hit / n, 2),
        )


score_rule("pool_unless_only_witness_matches_local", rule_pool_default)
score_rule("max_local_f1_tie_pool", rule_max_local)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


print("FEATURE MEANS on far exclusive wins")
for dataset in ("hotpotqa", "2wiki"):
    buckets = defaultdict(lambda: defaultdict(list))
    for key, methods in rows.items():
        if key[0] != dataset:
            continue
        pool = methods["centralized"]
        cover = methods["epirag_cover"]
        pa, ca = canon(pool), canon(cover)
        if token_f1(pa, ca) >= 0.5:
            continue
        pok = exact_match(pa, pool["gold"])
        cok = exact_match(ca, cover["gold"])
        if pok and not cok:
            label = "only_pool"
        elif cok and not pok:
            label = "only_cover"
        else:
            continue
        buckets[label]["margin"].append(float(cover["margin"] or 0.0))
        buckets[label]["nclaims"].append(len(cover["claims"] or []))
        beliefs = cover["beliefs"] or [0.0]
        buckets[label]["top"].append(max(beliefs))
        buckets[label]["triggered"].append(1.0 if cover["triggered"] else 0.0)
        buckets[label]["pconf"].append(float(pool["confidence"] or 0.0))
        buckets[label]["cconf"].append(float(cover["confidence"] or 0.0))
        gate = methods.get("epirag_gate") or {}
        buckets[label]["gate_witness"].append(1.0 if gate.get("decision") == "witness" else 0.0)
    for label, feats in buckets.items():
        print(
            dataset,
            label,
            "n", len(feats["margin"]),
            "margin", round(mean(feats["margin"]), 3),
            "nclaims", round(mean(feats["nclaims"]), 2),
            "top", round(mean(feats["top"]), 3),
            "trig", round(mean(feats["triggered"]), 2),
            "pconf", round(mean(feats["pconf"]), 4),
            "cconf", round(mean(feats["cconf"]), 4),
            "gate_w", round(mean(feats["gate_witness"]), 2),
        )


def rule_conf(threshold):
    def fn(pool, cover):
        pa, ca = canon(pool), canon(cover)
        if token_f1(pa, ca) >= 0.5:
            return pa
        gap = float(cover["confidence"] or 0.0) - float(pool["confidence"] or 0.0)
        if gap >= threshold:
            return pa
        return ca

    return fn


print("CONF GAP: disagree and witness_conf - pool_conf >= t -> pool, else witness; near stays pool")
for threshold in (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
    print("t", threshold)
    for dataset in ("hotpotqa", "2wiki"):
        n = hit = 0
        for key, methods in rows.items():
            if key[0] != dataset:
                continue
            chosen = rule_conf(threshold)(methods["centralized"], methods["epirag_cover"])
            n += 1
            hit += exact_match(chosen, methods["centralized"]["gold"])
        print(" ", dataset, round(100 * hit / n, 2))
