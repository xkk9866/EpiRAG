"""Development check: can one reader reconcile the pool and the witness?

Uses answers already stored on the forward block. Not a reported test.
The prompt below is fixed before any score from this script is read.
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch
from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import Generator, canonicalize_polar, evidence_prompt, parse_answer
from epirag.metrics import exact_match, token_f1
from epirag.run_eval import materialize, order_passages

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}


def union_pages(question: str, agents):
    pages = []
    seen = set()
    for group in agents:
        for passage in group:
            if passage.pid in seen:
                continue
            seen.add(passage.pid)
            pages.append(passage)
    return order_passages(question, pages)[:6]


def reconcile_prompt(question: str, passages, pool_answer: str, witness_answer: str) -> str:
    body = evidence_prompt(question, passages).split("Question:", 1)[1]
    return (
        "Answer the question using only the evidence. "
        "Two earlier readings disagree. "
        f"The reading of all pages says: {pool_answer}. "
        f"The reading of the conflict-filtered sentences says: {witness_answer}. "
        "Reply with the short answer span the evidence supports. "
        "You may copy one of those readings or a different span stated in the evidence. "
        "Do not explain, and do not use JSON.\n\n"
        f"Question:{body}"
    )


def canon(question: str, answer: str) -> str:
    if answer == "conflict":
        return answer
    return canonicalize_polar(question, answer)


def main():
    stored = defaultdict(dict)
    with open("experiments/results/records_forward.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS:
                continue
            if row["method"] not in ("centralized", "epirag_cover"):
                continue
            stored[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    instances = {}
    for name, data, budget in (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160)):
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="forward")
        for index in ids:
            inst = build_instance(data[index], name)
            instances[(name, inst.qid)] = inst
    jobs = []
    prompts = []
    for key, methods in stored.items():
        dataset, qid, condition = key
        inst = instances[(dataset, qid)]
        question = inst.question
        pool = canon(question, methods["centralized"]["answer"])
        cover = canon(question, methods["epirag_cover"]["answer"])
        agents, _archive = materialize(inst, condition)
        if pool == cover or token_f1(pool, cover) >= 0.999:
            jobs.append((dataset, question, inst.answer, pool, cover, "agree"))
            continue
        prompts.append(reconcile_prompt(question, union_pages(question, agents), pool, cover))
        jobs.append((dataset, question, inst.answer, pool, cover, "judge"))
    print(f"jobs {len(jobs)} judge prompts {len(prompts)}", flush=True)
    generator = Generator(
        "Qwen/Qwen2.5-7B-Instruct",
        "cuda:1",
        "experiments/cache/generations.jsonl",
        batch_size=4,
    )
    raw = generator.generate(prompts) if prompts else []
    cursor = 0
    totals = defaultdict(lambda: defaultdict(int))
    for dataset, question, gold, pool, cover, kind in jobs:
        if kind == "agree":
            chosen = pool
        else:
            chosen = canon(question, parse_answer(raw[cursor])[0])
            cursor += 1
        bucket = totals[dataset]
        bucket["n"] += 1
        bucket["judged"] += int(kind == "judge")
        pok = exact_match(pool, gold)
        cok = exact_match(cover, gold)
        jok = exact_match(chosen, gold)
        bucket["pool"] += pok
        bucket["cover"] += cok
        bucket["either"] += int(pok or cok)
        bucket["hit"] += jok
        if kind == "judge" and jok and not pok and not cok:
            bucket["third"] += 1
        if bucket["n"] % 200 == 0:
            print(dataset, bucket["n"], "hit", bucket["hit"], flush=True)
    for dataset, bucket in totals.items():
        n = bucket["n"]
        print(
            dataset,
            "n", n,
            "judge", round(100 * bucket["hit"] / n, 2),
            "pool", round(100 * bucket["pool"] / n, 2),
            "cover", round(100 * bucket["cover"] / n, 2),
            "either", round(100 * bucket["either"] / n, 2),
            "judged", bucket["judged"],
            "third_span_wins", bucket["third"],
            flush=True,
        )
    del generator
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
