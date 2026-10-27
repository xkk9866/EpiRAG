"""Development: majority of pool, witness, and graph. Not a reported test."""

import json
from collections import Counter, defaultdict

from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
FILES = {
    "advance": "experiments/results/records_advance.jsonl",
    "forward": "experiments/results/records_forward.jsonl",
}


def canon(row):
    if row["answer"] == "conflict":
        return row["answer"]
    return canonicalize_polar(row["question"], row["answer"])


def majority(answers, tie):
    counts = Counter(answers)
    best = max(counts.values())
    winners = [ans for ans, count in counts.items() if count == best]
    if len(winners) == 1:
        return winners[0]
    return tie


for name, path in FILES.items():
    rows = defaultdict(dict)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS:
                continue
            if row["method"] not in ("centralized", "epirag_cover", "epirag_graph"):
                continue
            rows[(row["dataset"], row["condition"], row["qid"])][row["method"]] = canon(row)
            rows[(row["dataset"], row["condition"], row["qid"])]["gold"] = row["gold"]
    print("\n", name)
    for dataset in ("hotpotqa", "2wiki"):
        n = 0
        base = Counter()
        voted = Counter()
        for key, item in rows.items():
            if key[0] != dataset:
                continue
            n += 1
            pool, cover, graph = item["centralized"], item["epirag_cover"], item["epirag_graph"]
            gold = item["gold"]
            for label, ans in (("pool", pool), ("cover", cover), ("graph", graph)):
                base[label] += exact_match(ans, gold)
            for tie_name, tie in (("tie_cover", cover), ("tie_pool", pool), ("tie_graph", graph)):
                voted[tie_name] += exact_match(majority([pool, cover, graph], tie), gold)
        print(
            dataset,
            "n", n,
            "pool", round(100 * base["pool"] / n, 2),
            "cover", round(100 * base["cover"] / n, 2),
            "graph", round(100 * base["graph"] / n, 2),
            {k: round(100 * v / n, 2) for k, v in voted.items()},
        )
