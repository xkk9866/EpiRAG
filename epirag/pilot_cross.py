"""Development check: can joint-evidence support choose the better reader?

The holdout block already scored the pool and the witness bundle. This script
does not generate new answers and is not a reported test. It only asks whether
a frozen comparison, cross_support, recovers exact matches that neither reader
has alone.
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch
from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import NLI, bundle_support, canonicalize_polar, joint_snippets, select_cross
from epirag.metrics import exact_match, token_f1
from epirag.run_eval import materialize

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}


def canon(question: str, answer: str) -> str:
    if answer == "conflict":
        return answer
    return canonicalize_polar(question, answer)


def load_answers():
    rows = defaultdict(dict)
    with open("experiments/results/records_holdout.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS:
                continue
            if row["method"] not in ("centralized", "epirag_cover"):
                continue
            rows[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
    return rows


def main():
    stored = load_answers()
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    instances = {}
    for name, data, budget in (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160)):
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="holdout")
        for index in ids:
            inst = build_instance(data[index], name)
            instances[(name, inst.qid)] = inst
    print(f"instances {len(instances)} stored keys {len(stored)}", flush=True)
    nli = NLI("cross-encoder/nli-deberta-v3-base", "cuda:0")
    totals = defaultdict(lambda: defaultdict(int))
    for key, methods in stored.items():
        dataset, qid, condition = key
        inst = instances[(dataset, qid)]
        pool = methods["centralized"]
        cover = methods["epirag_cover"]
        agents, _archive = materialize(inst, condition)
        snippets = joint_snippets(agents)
        question = inst.question
        pool_answer = canon(question, pool["answer"])
        cover_answer = canon(question, cover["answer"])
        chosen, decision = select_cross(nli, question, snippets, pool_answer, cover_answer)
        if bundle_support(nli, question, snippets, cover_answer) > bundle_support(
            nli, question, snippets, pool_answer
        ) and token_f1(pool_answer, cover_answer) < 0.999:
            mean_choice = cover_answer
        else:
            mean_choice = pool_answer
        gold = inst.answer
        bucket = totals[dataset]
        bucket["n"] += 1
        bucket["pool_em"] += exact_match(pool_answer, gold)
        bucket["cover_em"] += exact_match(cover_answer, gold)
        bucket["stated"] += exact_match(chosen, gold)
        bucket["mean"] += exact_match(mean_choice, gold)
        bucket[decision] += 1
        if bucket["n"] % 100 == 0:
            print(dataset, bucket["n"], "stated", bucket["stated"], "pool", bucket["pool_em"], flush=True)
    for dataset, bucket in totals.items():
        n = bucket["n"]
        print(
            dataset,
            "n", n,
            "stated", round(100 * bucket["stated"] / n, 2),
            "mean", round(100 * bucket["mean"] / n, 2),
            "pool", round(100 * bucket["pool_em"] / n, 2),
            "cover", round(100 * bucket["cover_em"] / n, 2),
            "agree", bucket["agree"],
            "pool_pick", bucket["pool"],
            "witness_pick", bucket["witness"],
            flush=True,
        )
    del nli
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
