"""Scores for the final block. The test families below were fixed before scoring.

Family A, 7B, Holm over 8 tests. Each test is stress-pooled exact match
on one dataset. Stress is asymmetry, corruption, temporal, and incomplete.
Contrasts: epirag_bridge against epirag_flat, debate, centralized, epirag_graph.

Family B, 3B, the same 8 tests, Holm inside this family only.

Family C, 7B collusion exact match, Holm over 4 tests:
epirag_bridge against epirag_cover and against centralized, on each dataset.

Family D, 7B, Holm over 2 tests: epirag_bridge against epirag_indep,
stress-pooled exact match on each dataset.

Clean is reported and is not in a family. Thresholds are not fit on this block.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, token_f1
from epirag.paired_tests import binom_two_sided, load

ROOT = Path(__file__).resolve().parents[1]
STRESS = ["asymmetry", "corruption", "temporal", "incomplete"]
CELLS = ["clean", "asymmetry", "corruption", "temporal", "incomplete", "collusion"]
DATASETS = ("hotpotqa", "2wiki")
FAMILY_A = [
    ("epirag_bridge", "epirag_flat"),
    ("epirag_bridge", "debate"),
    ("epirag_bridge", "centralized"),
    ("epirag_bridge", "epirag_graph"),
]
FAMILY_C = [
    ("epirag_bridge", "epirag_cover"),
    ("epirag_bridge", "centralized"),
]
FAMILY_D = [
    ("epirag_bridge", "epirag_indep"),
]


def fair_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        answer = row["answer"]
        if answer != "conflict":
            answer = canonicalize_polar(row["question"], answer)
        grouped[(row["dataset"], row["condition"], row["qid"])][row["method"]] = {
            "em": exact_match(answer, row["gold"]),
            "f1": token_f1(answer, row["gold"]),
            "tokens": row.get("comm_tokens") or 0,
            "reader_tokens": row.get("reader_tokens") or 0,
            "local_tokens": row.get("local_tokens") or 0,
            "nli_pairs": row.get("nli_pairs") or 0,
            "triggered": bool(row.get("triggered")),
            "decision": row.get("decision") or "",
        }
    return grouped


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def pooled(grouped, dataset, conditions, left, right):
    both = []
    for condition in conditions:
        for key, methods in grouped.items():
            if key[0] != dataset or key[1] != condition:
                continue
            if left in methods and right in methods:
                both.append(methods)
    if not both:
        return None
    left_only = sum(item[left]["em"] > item[right]["em"] for item in both)
    right_only = sum(item[right]["em"] > item[left]["em"] for item in both)
    discordant = left_only + right_only
    return {
        "n": len(both),
        "em_l": 100 * sum(item[left]["em"] for item in both) / len(both),
        "em_r": 100 * sum(item[right]["em"] for item in both) / len(both),
        "f1_l": 100 * sum(item[left]["f1"] for item in both) / len(both),
        "f1_r": 100 * sum(item[right]["f1"] for item in both) / len(both),
        "left_only": left_only,
        "right_only": right_only,
        "p": binom_two_sided(left_only, discordant),
    }


def cells(grouped, methods):
    for dataset in DATASETS:
        print(f"\n== {dataset}")
        for condition in CELLS:
            parts = []
            for method in methods:
                items = [
                    methods_at[method]
                    for key, methods_at in grouped.items()
                    if key[0] == dataset and key[1] == condition and method in methods_at
                ]
                if not items:
                    parts.append(f"{method}=NA")
                    continue
                em = 100 * sum(item["em"] for item in items) / len(items)
                f1 = 100 * sum(item["f1"] for item in items) / len(items)
                tok = sum(item["tokens"] for item in items) / len(items)
                parts.append(f"{method} {em:.1f}/{f1:.1f} ctok {tok:.0f}")
            print(condition, " | ".join(parts))


def decision_rates(rows):
    buckets = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["method"] not in ("epirag_indep", "epirag_bridge"):
            continue
        buckets[(row["method"], row["dataset"], row["condition"])][row.get("decision") or ""] += 1
    print("\n-- decisions")
    for key in sorted(buckets):
        total = sum(buckets[key].values())
        parts = [f"{name} {count}/{total}" for name, count in sorted(buckets[key].items())]
        print(key, " ".join(parts))


def trigger_rates(rows):
    print("\n-- trigger rates")
    for method in ("epirag_bridge", "epirag_indep", "epirag_indep_always", "epirag_indep_never", "epirag_cover"):
        for dataset in DATASETS:
            for condition in CELLS:
                items = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["dataset"] == dataset
                    and row["condition"] == condition
                ]
                if not items:
                    continue
                rate = 100 * sum(bool(row.get("triggered")) for row in items) / len(items)
                print(f"{method:22} {dataset:9} {condition:11} {rate:.1f}")


def cost_table(grouped):
    print("\n-- stress mean cost")
    methods = [
        "centralized",
        "debate",
        "collab",
        "epirag_cover",
        "epirag_flat",
        "epirag_indep",
        "epirag_indep_always",
        "epirag_indep_never",
        "epirag_graph",
        "epirag_bridge",
        "epirag_oracle",
        "epirag_gate",
        "epirag_mark",
        "epirag_chain",
        "epirag_exec",
        "epirag_hop",
        "epirag_derive",
    ]
    for dataset in DATASETS:
        print(dataset)
        for method in methods:
            items = [
                methods_at[method]
                for key, methods_at in grouped.items()
                if key[0] == dataset and key[1] in STRESS and method in methods_at
            ]
            if not items:
                continue
            n = len(items)
            em = 100 * sum(item["em"] for item in items) / n
            f1 = 100 * sum(item["f1"] for item in items) / n
            comm = sum(item["tokens"] for item in items) / n
            reader = sum(item["reader_tokens"] for item in items) / n
            local = sum(item["local_tokens"] for item in items) / n
            pairs = sum(item["nli_pairs"] for item in items) / n
            print(
                f"  {method:22} EM {em:.1f} F1 {f1:.1f} comm {comm:.0f} "
                f"reader {reader:.0f} local {local:.0f} nli {pairs:.0f}"
            )


def family(grouped, contrasts, conditions, label):
    stats = []
    for dataset in DATASETS:
        for left, right in contrasts:
            item = pooled(grouped, dataset, conditions, left, right)
            if item is None:
                continue
            item["dataset"] = dataset
            item["left"] = left
            item["right"] = right
            stats.append(item)
    adjusted = holm([item["p"] for item in stats]) if stats else []
    print(f"\n-- {label}")
    for item, p_holm in zip(stats, adjusted):
        print(
            f"{item['dataset']:9} {item['left']:22} vs {item['right']:16} "
            f"EM {item['em_l']:.1f}/{item['em_r']:.1f} F1 {item['f1_l']:.1f}/{item['f1_r']:.1f} "
            f"only {item['left_only']}/{item['right_only']} "
            f"p={item['p']:.4g} holm={p_holm:.4g} n={item['n']}"
        )


def report(path: Path):
    rows = load(path)
    print(f"\n##### {path} rows {len(rows)}")
    grouped = fair_rows(rows)
    methods = [
        "centralized",
        "debate",
        "collab",
        "epirag_cover",
        "epirag_flat",
        "epirag_indep",
        "epirag_indep_always",
        "epirag_indep_never",
        "epirag_graph",
        "epirag_bridge",
        "epirag_oracle",
        "epirag_gate",
    ]
    cells(grouped, methods)
    family(grouped, FAMILY_A, STRESS, "family A/B stress")
    family(grouped, FAMILY_D, STRESS, "family D bridge vs direct")
    family(grouped, FAMILY_C, ["collusion"], "family C collusion")
    decision_rates(rows)
    trigger_rates(rows)
    cost_table(grouped)


if __name__ == "__main__":
    report(ROOT / "experiments" / "results" / "records_final.jsonl")
    other = ROOT / "experiments" / "results" / "records_final_3b.jsonl"
    if other.exists() and other.stat().st_size:
        report(other)
