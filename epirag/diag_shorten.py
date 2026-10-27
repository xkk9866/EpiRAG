"""Development: replace a prediction by the shortest local claim it contains."""

import json
from collections import defaultdict

from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, normalize_answer, token_f1

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
FILES = {
    "advance": "experiments/results/records_advance.jsonl",
    "forward": "experiments/results/records_forward.jsonl",
}


def shorten(answer: str, claims: list[str]) -> str:
    host = normalize_answer(answer)
    if not host:
        return answer
    contained = []
    for claim in claims:
        key = normalize_answer(claim)
        if key and key in host and key != host:
            contained.append(claim)
    if not contained:
        return answer
    return min(contained, key=lambda claim: (len(normalize_answer(claim).split()), len(claim)))


def main():
    for name, path in FILES.items():
        print("==", name)
        grouped = defaultdict(dict)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["condition"] not in STRESS or row["method"] not in {"epirag_cover", "centralized"}:
                    continue
                grouped[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
        stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
        for methods in grouped.values():
            if "epirag_cover" not in methods:
                continue
            cover = methods["epirag_cover"]
            question = cover["question"]
            gold = cover["gold"]
            claims = cover.get("claims") or []
            for method in ("epirag_cover", "centralized"):
                if method not in methods:
                    continue
                raw = canonicalize_polar(question, methods[method]["answer"])
                short = canonicalize_polar(question, shorten(raw, claims))
                bucket = stats[cover["dataset"]]
                for label, value in (
                    (method + "_em", exact_match(raw, gold)),
                    (method + "_f1", token_f1(raw, gold)),
                    (method + "_short_em", exact_match(short, gold)),
                    (method + "_short_f1", token_f1(short, gold)),
                ):
                    bucket[label][0] += value
                    bucket[label][2] += 1
        for dataset, bucket in stats.items():
            n = bucket["epirag_cover_em"][2]
            print(dataset, "n", n)
            for label, (value, _unused, total) in bucket.items():
                print(f"  {label:22} {100 * value / total:.2f}" if total else label)


if __name__ == "__main__":
    main()
