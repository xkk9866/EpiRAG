"""Development scores for claim-consistency repairs. Not a reported test."""

import json
from collections import defaultdict

from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, token_f1

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
FILES = {
    "advance": "experiments/results/records_advance.jsonl",
    "forward": "experiments/results/records_forward.jsonl",
}


def best_f1(answer: str, claims: list[str]) -> float:
    if not claims:
        return 0.0
    return max(token_f1(answer, claim) for claim in claims)


def majority(answers: list[str], question: str) -> str:
    keyed = []
    for answer in answers:
        text = canonicalize_polar(question, answer or "")
        key = " ".join(text.lower().split())
        if key and key not in {"unknown", "none", "[]"}:
            keyed.append((key, text))
    if not keyed:
        return ""
    counts = defaultdict(int)
    for key, _text in keyed:
        counts[key] += 1
    best = max(counts.values())
    for key, text in keyed:
        if counts[key] == best:
            return text
    return ""


def main():
    for name, path in FILES.items():
        grouped = defaultdict(dict)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["condition"] not in STRESS:
                    continue
                if row["method"] not in {"epirag_cover", "centralized", "epirag_flat", "epirag_graph"}:
                    continue
                grouped[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
        stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        alien = defaultdict(lambda: [0, 0, 0, 0])
        for key, methods in grouped.items():
            if "epirag_cover" not in methods:
                continue
            cover = methods["epirag_cover"]
            question = cover["question"]
            gold = cover["gold"]
            dataset = cover["dataset"]
            claims = cover.get("claims") or []
            cover_ans = canonicalize_polar(question, cover["answer"])
            pool_ans = canonicalize_polar(question, methods["centralized"]["answer"]) if "centralized" in methods else ""
            flat_ans = canonicalize_polar(question, methods["epirag_flat"]["answer"]) if "epirag_flat" in methods else ""
            graph_ans = canonicalize_polar(question, methods["epirag_graph"]["answer"]) if "epirag_graph" in methods else ""
            local = majority(cover.get("local_answers") or [], question)
            cover_fit = best_f1(cover_ans, claims)
            pool_fit = best_f1(pool_ans, claims)
            snapped = cover_ans
            if cover_fit < 0.5 and claims:
                snapped = max(claims, key=lambda claim: token_f1(cover_ans, claim))
                if token_f1(cover_ans, snapped) <= 0:
                    snapped = local or cover_ans
            rescue_pool = pool_ans if cover_fit < 0.5 and pool_fit >= 0.5 else cover_ans
            rescue_graph = graph_ans if cover_fit < 0.5 and best_f1(graph_ans, claims) >= 0.5 else cover_ans
            rescue_flat = flat_ans if cover_fit < 0.5 and best_f1(flat_ans, claims) >= 0.5 else cover_ans
            preds = {
                "cover": cover_ans,
                "pool": pool_ans,
                "flat": flat_ans,
                "graph": graph_ans,
                "local_majority": local or cover_ans,
                "snap": snapped,
                "rescue_pool": rescue_pool,
                "rescue_graph": rescue_graph,
                "rescue_flat": rescue_flat,
            }
            for label, pred in preds.items():
                stats[dataset][label][0] += exact_match(pred, gold)
                stats[dataset][label][1] += 1
            if cover_fit < 0.5:
                alien[dataset][0] += 1
                alien[dataset][1] += exact_match(cover_ans, gold)
                alien[dataset][2] += exact_match(pool_ans, gold)
                alien[dataset][3] += exact_match(local or "", gold)
        print("==", name)
        for dataset, rows in stats.items():
            n = rows["cover"][1]
            print(dataset, "n", n)
            for label, (hit, total) in rows.items():
                print(f"  {label:16} {100 * hit / total:.2f}")
            a_n, a_cover, a_pool, a_local = alien[dataset]
            print(
                f"  alien n {a_n} cover {100 * a_cover / a_n:.1f} "
                f"pool {100 * a_pool / a_n:.1f} local {100 * a_local / a_n:.1f}"
                if a_n
                else "  alien n 0"
            )


if __name__ == "__main__":
    main()
