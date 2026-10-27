"""Collapse 2Wiki compositional and inference questions to templates."""

import json
import re
from collections import Counter

counts = Counter()
with open("experiments/results/records_prospect.jsonl", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if row["method"] != "epirag_cover" or row["condition"] != "clean":
            continue
        if row["qtype"] not in {"compositional", "inference"}:
            continue
        text = row["question"].rstrip("?")
        text = re.sub(r"\b(film|song|album) .+$", r"\1 WORK", text, count=1)
        text = re.sub(r"\bof film .+", "of film WORK", text)
        text = re.sub(r"\bof song .+", "of song WORK", text)
        text = re.sub(r"published .+", "published WORK", text)
        text = re.sub(r"^Who is the (maternal|paternal) \S+ of .+", r"Who is the \1 REL of NAME", text)
        text = re.sub(r"^Who is .+?'s .+", "Who is NAME's REL", text)
        text = re.sub(r" of [A-Z].+", " of NAME", text)
        counts[(row["qtype"], text)] += 1
for key, count in counts.most_common():
    print(f"{count:3} {key[0]:14} {key[1]}")
