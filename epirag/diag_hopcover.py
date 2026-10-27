"""Show which advance questions the hop grammar covers. No reader."""

import json
from collections import Counter

from epirag.hop import decompose

PATH = "experiments/results/records_advance.jsonl"
seen = set()
covered = Counter()
missed = Counter()
with open(PATH, encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if row["method"] != "epirag_cover" or row["condition"] != "clean":
            continue
        if row["qtype"] not in {"compositional", "inference"}:
            continue
        key = (row["dataset"], row["question"])
        if key in seen:
            continue
        seen.add(key)
        hops = decompose(row["question"])
        bucket = covered if hops else missed
        bucket[row["dataset"]] += 1
        if hops and covered[row["dataset"]] <= 6:
            print("HIT", hops["hop1"], "||", hops["hop2"])
        if not hops and missed[row["dataset"]] <= 8:
            print("MISS", row["qtype"], row["question"])
print("covered", dict(covered))
print("missed", dict(missed))
