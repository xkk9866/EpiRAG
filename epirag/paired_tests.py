"""Paired comparisons and polar-map sensitivity. No GPU."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, expected_calibration_error, token_f1

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["clean", "asymmetry", "corruption", "temporal", "incomplete"]
STRESS = ["asymmetry", "corruption", "temporal", "incomplete"]


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    # Two-sided exact test of p=0.5.
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def paired(rows, method_a: str, method_b: str, dataset: str, condition: str):
    by_q = defaultdict(dict)
    for row in rows:
        if row["dataset"] != dataset or row["condition"] != condition:
            continue
        if row["method"] not in (method_a, method_b):
            continue
        answer = canonicalize_polar(row["question"], row["answer"])
        by_q[row["qid"]][row["method"]] = (
            exact_match(answer, row["gold"]),
            token_f1(answer, row["gold"]),
        )
    both = [item for item in by_q.values() if method_a in item and method_b in item]
    if not both:
        return None
    a_only = sum(1 for item in both if item[method_a][0] > item[method_b][0])
    b_only = sum(1 for item in both if item[method_b][0] > item[method_a][0])
    em_a = 100 * sum(item[method_a][0] for item in both) / len(both)
    em_b = 100 * sum(item[method_b][0] for item in both) / len(both)
    f1_a = 100 * sum(item[method_a][1] for item in both) / len(both)
    f1_b = 100 * sum(item[method_b][1] for item in both) / len(both)
    discordant = a_only + b_only
    return {
        "n": len(both),
        "em_a": em_a,
        "em_b": em_b,
        "f1_a": f1_a,
        "f1_b": f1_b,
        "a_only": a_only,
        "b_only": b_only,
        "p": binom_two_sided(a_only, discordant),
    }


def polar_changes(rows, method: str):
    changed = 0
    total = 0
    em_raw = 0.0
    em_fair = 0.0
    f1_raw = 0.0
    f1_fair = 0.0
    conf = []
    raw_correct = []
    fair_correct = []
    for row in rows:
        if row["method"] != method:
            continue
        total += 1
        fair = canonicalize_polar(row["question"], row["answer"])
        if fair != row["answer"]:
            changed += 1
        em_raw += exact_match(row["answer"], row["gold"])
        em_fair += exact_match(fair, row["gold"])
        f1_raw += token_f1(row["answer"], row["gold"])
        f1_fair += token_f1(fair, row["gold"])
        conf.append(float(row["confidence"]))
        raw_correct.append(exact_match(row["answer"], row["gold"]))
        fair_correct.append(exact_match(fair, row["gold"]))
    if total == 0:
        return None
    return {
        "n": total,
        "changed": changed,
        "em_raw": 100 * em_raw / total,
        "em_fair": 100 * em_fair / total,
        "f1_raw": 100 * f1_raw / total,
        "f1_fair": 100 * f1_fair / total,
        "ece_raw": expected_calibration_error(conf, raw_correct),
        "ece_fair": expected_calibration_error(conf, fair_correct),
    }


def cell_table(rows, methods):
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] not in methods:
            continue
        answer = canonicalize_polar(row["question"], row["answer"])
        grouped[(row["dataset"], row["condition"], row["method"])].append(
            (exact_match(answer, row["gold"]), token_f1(answer, row["gold"]), row["comm_tokens"])
        )
    for dataset in ("hotpotqa", "2wiki"):
        print(f"\n== {dataset}")
        for condition in CONDITIONS:
            parts = []
            for method in methods:
                items = grouped.get((dataset, condition, method), [])
                if not items:
                    parts.append(f"{method}=NA")
                    continue
                em = 100 * sum(x[0] for x in items) / len(items)
                f1 = 100 * sum(x[1] for x in items) / len(items)
                tok = sum(x[2] for x in items) / len(items)
                parts.append(f"{method} {em:.1f}/{f1:.1f} tok {tok:.0f}")
            print(condition, " | ".join(parts))


def report(path: Path, pairs):
    rows = load(path)
    print(f"\n##### {path.name} rows {len(rows)}")
    methods = [
        "independent",
        "centralized",
        "debate",
        "collab",
        "epirag_cover",
        "epirag_flat",
        "epirag_drop",
        "epirag_cover_noret",
        "epirag",
    ]
    cell_table(rows, methods)
    print("\n-- paired fair EM")
    for dataset in ("hotpotqa", "2wiki"):
        for condition in CONDITIONS:
            for left, right in pairs:
                stats = paired(rows, left, right, dataset, condition)
                if stats is None:
                    continue
                print(
                    f"{dataset:9} {condition:11} {left:18} vs {right:18} "
                    f"EM {stats['em_a']:.1f}/{stats['em_b']:.1f} "
                    f"F1 {stats['f1_a']:.1f}/{stats['f1_b']:.1f} "
                    f"only {stats['a_only']}/{stats['b_only']} p={stats['p']:.4f} n={stats['n']}"
                )
    print("\n-- polar map and ECE")
    for method in ("epirag_cover", "debate", "centralized", "epirag_flat", "epirag_drop"):
        stats = polar_changes(rows, method)
        if stats is None:
            continue
        print(
            f"{method:18} n={stats['n']} changed={stats['changed']} "
            f"EM {stats['em_raw']:.1f}->{stats['em_fair']:.1f} "
            f"F1 {stats['f1_raw']:.1f}->{stats['f1_fair']:.1f} "
            f"ECE raw {stats['ece_raw']:.3f} fair-label {stats['ece_fair']:.3f}"
        )


if __name__ == "__main__":
    pairs = [
        ("epirag_cover", "epirag_flat"),
        ("epirag_cover", "debate"),
        ("epirag_cover", "centralized"),
        ("epirag_cover", "epirag_drop"),
        ("epirag_flat", "epirag_drop"),
        ("epirag_cover", "epirag_cover_noret"),
    ]
    report(ROOT / "experiments/results/records_v2.jsonl", pairs)
    confirm = ROOT / "experiments/results/records_confirm.jsonl"
    if confirm.exists() and confirm.stat().st_size > 0:
        report(confirm, pairs)
