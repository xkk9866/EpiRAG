"""Among misses where the gold span is on the page, how close is the emitted string?"""

import json
from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, normalize_answer, token_f1
from epirag.run_eval import materialize

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}


def main():
    stored = defaultdict(dict)
    with open("experiments/results/records_advance.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS:
                continue
            if row["method"] not in ("centralized", "epirag_cover", "epirag_graph"):
                continue
            stored[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    instances = {}
    for name, data, budget in (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160)):
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="advance")
        for index in ids:
            inst = build_instance(data[index], name)
            instances[(name, inst.qid)] = inst
    for dataset in ("hotpotqa", "2wiki"):
        print(dataset)
        for method in ("centralized", "epirag_cover", "epirag_graph"):
            near = wrong = n = 0
            for key, methods in stored.items():
                if key[0] != dataset:
                    continue
                inst = instances[(dataset, key[1])]
                agents, _archive = materialize(inst, key[2])
                blob = normalize_answer(" ".join(p.text for group in agents for p in group))
                gold_norm = normalize_answer(inst.answer)
                if not gold_norm or gold_norm not in blob:
                    continue
                answer = methods[method]["answer"]
                if answer != "conflict":
                    answer = canonicalize_polar(inst.question, answer)
                if exact_match(answer, inst.answer):
                    continue
                n += 1
                if token_f1(answer, inst.answer) >= 0.5:
                    near += 1
                else:
                    wrong += 1
            print(f"  {method:16} in-page misses {n} near {near} other {wrong}")


if __name__ == "__main__":
    main()
