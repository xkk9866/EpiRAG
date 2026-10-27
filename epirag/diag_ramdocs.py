"""Ceiling of a document-count filter on RAMDocs, using annotated answers.

Not a system result. It only asks whether minority-answer deletion can
satisfy the published strict criterion before any reader is run.
"""

import json
from collections import defaultdict

from epirag.metrics import normalize_answer, token_f1


def strict_hit(prediction_parts, golds, wrongs):
    pred = " ".join(normalize_answer(part) for part in prediction_parts)
    gold_ok = all(normalize_answer(gold) and normalize_answer(gold) in pred for gold in golds)
    wrong_ok = all(not normalize_answer(wrong) or normalize_answer(wrong) not in pred for wrong in wrongs)
    return gold_ok and wrong_ok


def clusters(pairs):
    groups = []
    for answer, kind in pairs:
        placed = False
        for group in groups:
            if token_f1(answer, group[0][0]) >= 0.5:
                group.append((answer, kind))
                placed = True
                break
        if not placed:
            groups.append([(answer, kind)])
    return groups


def main():
    rows = [json.loads(line) for line in open("experiments/data/RAMDocs_test.jsonl", encoding="utf-8")]
    all_docs = minority = correct_only = 0
    for row in rows:
        pairs = []
        for doc in row["documents"]:
            answer = str(doc.get("answer") or "")
            if normalize_answer(answer) in {"", "unknown"}:
                continue
            pairs.append((answer, doc["type"]))
        kept_all = [answer for answer, _kind in pairs]
        groups = clusters(pairs)
        if not groups:
            kept_min = []
        else:
            sizes = [len(group) for group in groups]
            peak = max(sizes)
            kept_min = []
            for group, size in zip(groups, sizes):
                if size == peak:
                    kept_min.append(group[0][0])
        all_docs += strict_hit(kept_all, row["gold_answers"], row["wrong_answers"])
        minority += strict_hit(kept_min, row["gold_answers"], row["wrong_answers"])
        correct_only += strict_hit(
            [str(doc.get("answer") or "") for doc in row["documents"] if doc["type"] == "correct"],
            row["gold_answers"],
            row["wrong_answers"],
        )
    n = len(rows)
    print("n", n)
    print("emit every non-noise annotated answer", round(100 * all_docs / n, 2))
    print("keep only largest answer groups", round(100 * minority / n, 2))
    print("oracle correct-type answers", round(100 * correct_only / n, 2))


if __name__ == "__main__":
    main()
