"""Development pilot for relational hops on the advance block.

Not a reported test. A missing hop keeps the stored witness answer.
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch
from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.diag_chainsearch import pages_of
from epirag.engine import Generator, canonicalize_polar
from epirag.hop import decompose, resolve_hops
from epirag.metrics import exact_match

STRESS = ("asymmetry", "corruption", "temporal", "incomplete")


def main() -> None:
    print("loading generator", flush=True)
    generator = Generator(
        "Qwen/Qwen2.5-7B-Instruct",
        "cuda:1",
        "experiments/cache/generations.jsonl",
        batch_size=2,
    )
    print("loading data", flush=True)
    stored = {}
    with open("experiments/results/records_advance.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS or row["method"] != "epirag_cover":
                continue
            stored[(row["dataset"], row["qid"], row["condition"])] = row
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    jobs = []
    for name, data, budget in (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160)):
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="advance")
        for index in ids:
            instance = build_instance(data[index], name)
            if decompose(instance.question) is None:
                continue
            for condition in STRESS:
                key = (name, instance.qid, condition)
                if key not in stored:
                    continue
                jobs.append(
                    {
                        "dataset": name,
                        "qid": instance.qid,
                        "condition": condition,
                        "question": instance.question,
                        "gold": instance.answer,
                        "pages": pages_of(instance, condition),
                        "cover": stored[key]["answer"],
                        "qtype": instance.qtype,
                    }
                )
    print(f"hop jobs {len(jobs)}", flush=True)
    packed = resolve_hops(generator.generate, [(job["question"], job["pages"]) for job in jobs])
    answers = [item["answer"] for item in packed]
    totals = defaultdict(lambda: [0, 0, 0, 0])
    overall = defaultdict(lambda: [0, 0, 0])
    for row in stored.values():
        hit = exact_match(canonicalize_polar(row["question"], row["answer"]), row["gold"])
        overall[row["dataset"]][0] += 1
        overall[row["dataset"]][1] += hit
        overall[row["dataset"]][2] += hit
    shown = defaultdict(int)
    for job, structural in zip(jobs, answers):
        cover = canonicalize_polar(job["question"], job["cover"])
        pred = structural or cover
        bucket = totals[job["dataset"]]
        bucket[0] += 1
        bucket[1] += exact_match(cover, job["gold"])
        bucket[2] += exact_match(pred, job["gold"])
        if structural:
            bucket[3] += 1
            hit = exact_match(structural, job["gold"])
            totals[job["dataset"] + ":fire"][0] += 1
            totals[job["dataset"] + ":fire"][1] += hit
            if not hit and shown[job["dataset"]] < 6:
                shown[job["dataset"]] += 1
                print("MISS", job["qtype"], "gold=", job["gold"], "pred=", structural, flush=True)
                print(" ", job["question"][:150], flush=True)
        overall[job["dataset"]][2] += exact_match(pred, job["gold"]) - exact_match(cover, job["gold"])
    for dataset, (n, cover_hit, hybrid_hit, fired) in list(totals.items()):
        if dataset.endswith(":fire"):
            continue
        fire = totals[dataset + ":fire"]
        precision = 100 * fire[1] / fire[0] if fire[0] else 0.0
        all_n, all_cover, all_hybrid = overall[dataset]
        print(
            f"{dataset} hop-questions n {n} cover {100 * cover_hit / n:.2f} "
            f"hybrid {100 * hybrid_hit / n:.2f} fire {100 * fired / n:.1f} precision {precision:.1f}",
            flush=True,
        )
        print(
            f"{dataset} stress n {all_n} cover {100 * all_cover / all_n:.2f} "
            f"hybrid {100 * all_hybrid / all_n:.2f}",
            flush=True,
        )
    del generator
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
