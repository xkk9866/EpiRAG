"""Where stress errors sit: the gold span is in the pages, or it must be composed."""

import json
from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, normalize_answer
from epirag.run_eval import materialize

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
METHODS = ("centralized", "epirag_cover", "epirag_graph", "epirag_mark", "debate")


def main():
    stored = defaultdict(dict)
    with open("experiments/results/records_advance.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS or row["method"] not in METHODS:
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
        present = absent = 0
        hit = defaultdict(lambda: [0, 0])
        for key, methods in stored.items():
            if key[0] != dataset:
                continue
            inst = instances[(dataset, key[1])]
            agents, _archive = materialize(inst, key[2])
            blob = normalize_answer(" ".join(p.text for group in agents for p in group))
            gold_norm = normalize_answer(inst.answer)
            in_pages = bool(gold_norm) and gold_norm in blob
            if in_pages:
                present += 1
            else:
                absent += 1
            slot = 0 if in_pages else 1
            for method in METHODS:
                answer = methods[method]["answer"]
                if answer != "conflict":
                    answer = canonicalize_polar(inst.question, answer)
                hit[method][slot] += exact_match(answer, inst.answer)
        print(dataset, "present", present, "absent", absent)
        for method in METHODS:
            p = 100 * hit[method][0] / present if present else 0
            a = 100 * hit[method][1] / absent if absent else 0
            print(f"  {method:16} in_pages {p:.1f} composed {a:.1f}")


if __name__ == "__main__":
    main()
