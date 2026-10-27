"""Development: pick the reader whose stating sentence matches the question.

The rule does not use the gold answer. Exact match is only the score.
"""

import json
from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import QUESTION_STOP, canonicalize_polar, split_sentences
from epirag.metrics import exact_match, normalize_answer
from epirag.run_eval import materialize

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
FILES = {
    "advance": "experiments/results/records_advance.jsonl",
    "forward": "experiments/results/records_forward.jsonl",
}


def content_words(question: str, answers: list[str]) -> set[str]:
    blocked = set(QUESTION_STOP)
    for answer in answers:
        blocked.update(normalize_answer(answer).split())
    return {token for token in normalize_answer(question).split() if len(token) > 3 and token not in blocked}


def stated(sentence: str, answer: str) -> bool:
    gold = normalize_answer(answer).split()
    if not gold:
        return False
    have = set(normalize_answer(sentence).split())
    return sum(token in have for token in gold) / len(gold) >= 0.5


def support(answer: str, sentences: list[str], words: set[str]) -> int:
    if not normalize_answer(answer):
        return -1
    best = -1
    for sentence in sentences:
        if not stated(sentence, answer):
            continue
        have = set(normalize_answer(sentence).split())
        best = max(best, sum(token in have for token in words))
    return best


def main():
    rows = defaultdict(dict)
    for split, path in FILES.items():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["condition"] not in STRESS:
                    continue
                if row["method"] not in {"epirag_cover", "centralized", "epirag_graph"}:
                    continue
                rows[split][(row["dataset"], row["qid"], row["condition"], row["method"])] = row
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    specs = (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160))
    for split in FILES:
        print("==", split, flush=True)
        for name, data, budget in specs:
            _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block=split)
            instances = [build_instance(data[i], name) for i in ids]
            totals = defaultdict(lambda: [0, 0])
            for instance in instances:
                for condition in ("asymmetry", "corruption", "temporal", "incomplete"):
                    agents, _archive = materialize(instance, condition)
                    sentences = []
                    seen = set()
                    for group in agents:
                        for passage in group:
                            if passage.pid in seen:
                                continue
                            seen.add(passage.pid)
                            sentences.extend(split_sentences(passage.text))
                            sentences.append(passage.title + ". " + passage.text[:240])
                    key = (name, instance.qid, condition)
                    cover = rows[split].get(key + ("epirag_cover",))
                    pool = rows[split].get(key + ("centralized",))
                    graph = rows[split].get(key + ("epirag_graph",))
                    if not cover or not pool or not graph:
                        continue
                    question = instance.question
                    answers = {
                        "cover": canonicalize_polar(question, cover["answer"]),
                        "pool": canonicalize_polar(question, pool["answer"]),
                        "graph": canonicalize_polar(question, graph["answer"]),
                    }
                    words = content_words(question, list(answers.values()))
                    scores = {label: support(text, sentences, words) for label, text in answers.items()}
                    # Tie keeps the witness answer. Switch only on a strictly better relation match.
                    picked = "cover"
                    if scores["pool"] > scores["cover"] and scores["pool"] >= scores["graph"]:
                        picked = "pool"
                    elif scores["graph"] > scores["cover"] and scores["graph"] > scores["pool"]:
                        picked = "graph"
                    gold = instance.answer
                    for label, text in answers.items():
                        totals[label][0] += exact_match(text, gold)
                        totals[label][1] += 1
                    totals["relation"][0] += exact_match(answers[picked], gold)
                    totals["relation"][1] += 1
            n = totals["cover"][1]
            print(name, "n", n, flush=True)
            for label, (hit, total) in totals.items():
                print(f"  {label:10} {100 * hit / total:.2f}", flush=True)


if __name__ == "__main__":
    main()
