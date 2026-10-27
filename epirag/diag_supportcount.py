"""Development: keep the reading stated by more private views."""

import json
from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, normalize_answer
from epirag.run_eval import materialize

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
FILES = {
    "advance": ("experiments/results/records_advance.jsonl", "advance"),
    "forward": ("experiments/results/records_forward.jsonl", "forward"),
}


def canon(question, answer):
    if answer == "conflict":
        return answer
    return canonicalize_polar(question, answer)


def support_count(answer, agents):
    key = normalize_answer(answer)
    if not key:
        return 0
    count = 0
    for group in agents:
        blob = normalize_answer(" ".join(passage.text for passage in group))
        if key in blob:
            count += 1
    return count


def choose(pool, cover, agents, tie):
    ps = support_count(pool, agents)
    cs = support_count(cover, agents)
    if cs > ps:
        return cover
    if ps > cs:
        return pool
    return tie


def main():
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    for label, (path, block) in FILES.items():
        stored = defaultdict(dict)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["condition"] not in STRESS:
                    continue
                if row["method"] not in ("centralized", "epirag_cover"):
                    continue
                stored[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
        instances = {}
        budgets = {"hotpotqa": 200, "2wiki": 160}
        for name, data in (("hotpotqa", hotpot), ("2wiki", wiki)):
            _dev, ids = select_examples(data, name, budgets[name], n_dev=12, seed=7, block=block)
            for index in ids:
                inst = build_instance(data[index], name)
                instances[(name, inst.qid)] = inst
        print("\n", label)
        for dataset in ("hotpotqa", "2wiki"):
            n = pool_hit = cover_hit = tie_cover = tie_pool = 0
            for key, methods in stored.items():
                if key[0] != dataset:
                    continue
                inst = instances[(dataset, key[1])]
                agents, _archive = materialize(inst, key[2])
                pool = canon(inst.question, methods["centralized"]["answer"])
                cover = canon(inst.question, methods["epirag_cover"]["answer"])
                gold = inst.answer
                n += 1
                pool_hit += exact_match(pool, gold)
                cover_hit += exact_match(cover, gold)
                tie_cover += exact_match(choose(pool, cover, agents, cover), gold)
                tie_pool += exact_match(choose(pool, cover, agents, pool), gold)
            print(
                dataset,
                "n", n,
                "pool", round(100 * pool_hit / n, 2),
                "cover", round(100 * cover_hit / n, 2),
                "count_tie_cover", round(100 * tie_cover / n, 2),
                "count_tie_pool", round(100 * tie_pool / n, 2),
            )


if __name__ == "__main__":
    main()
