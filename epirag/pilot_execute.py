"""Development pilot: executed comparison against stored witness answers.

Uses the advance block only as a go/no-go. It does not write a result file
and it is not a reported test.
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch
from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.diag_chainsearch import pages_of
from epirag.engine import Generator, canonicalize_polar
from epirag.execute import (
    attribute,
    decide,
    director_prompt,
    direction_later,
    hop_prompt,
    parse_person,
    parse_year,
    plan,
    split_pair,
)
from epirag.metrics import exact_match

STRESS = ("asymmetry", "corruption", "temporal", "incomplete")
COVER_PATHS = {
    "advance": "experiments/results/records_advance.jsonl",
    "forward": "experiments/results/records_forward.jsonl",
}


def _check_plans() -> None:
    song = plan("Which song came out earlier, Can'T Remember To Forget You or Distant Lover?")
    film = plan("Which film has the director who was born later, South Of Algiers or Kabut Sutra Ungu?")
    born = plan("Who was born first Václav Havel or Penelope Fitzgerald?")
    assert song == {
        "left": "Can'T Remember To Forget You",
        "right": "Distant Lover",
        "kind": "year",
        "later": False,
    }, song
    assert film["kind"] == "director-born" and film["later"] is True, film
    assert film["left"] == "South Of Algiers" and film["right"] == "Kabut Sutra Ungu"
    assert born["left"] == "Václav Havel" and born["kind"] == "born" and born["later"] is False, born
    assert plan("What pop band features an Argentine actress?") is None
    assert split_pair("Are Yingkou and Fuding the same level?") is None
    assert attribute("Are they the same?") is None
    assert direction_later("Which came first?") is False
    print("plan checks passed", flush=True)


def load_cover(path: str) -> dict:
    stored = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS or row["method"] != "epirag_cover":
                continue
            stored[(row["dataset"], row["qid"], row["condition"])] = row
    return stored


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=("advance", "forward"), default="advance")
    args = parser.parse_args()
    _check_plans()
    print("loading generator", flush=True)
    generator = Generator(
        "Qwen/Qwen2.5-7B-Instruct",
        "cuda:1",
        "experiments/cache/generations.jsonl",
        batch_size=4,
    )
    print("loading data", flush=True)
    stored = load_cover(COVER_PATHS[args.slice])
    print("slice", args.slice, flush=True)
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    specs = (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160))
    jobs = []
    for name, data, budget in specs:
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block=args.slice)
        for index in ids:
            instance = build_instance(data[index], name)
            spec = plan(instance.question)
            if spec is None:
                continue
            for condition in STRESS:
                if (name, instance.qid, condition) not in stored:
                    continue
                pages = pages_of(instance, condition)
                jobs.append(
                    {
                        "dataset": name,
                        "qid": instance.qid,
                        "condition": condition,
                        "question": instance.question,
                        "gold": instance.answer,
                        "spec": spec,
                        "pages": pages,
                        "cover": stored[(name, instance.qid, condition)]["answer"],
                    }
                )
    print(f"comparative jobs {len(jobs)}", flush=True)
    first_prompts = []
    first_meta = []
    for job_index, job in enumerate(jobs):
        spec = job["spec"]
        kind = spec["kind"]
        for side in ("left", "right"):
            entity = spec[side]
            if kind.startswith("director"):
                prompt = director_prompt(entity, job["pages"])
                role = "director"
            else:
                prompt = hop_prompt(job["question"], entity, kind, job["pages"])
                role = "year"
            first_prompts.append(prompt or "")
            first_meta.append((job_index, side, role, prompt is not None))
    raw_first = generator.generate([prompt for prompt in first_prompts if prompt])
    cursor = 0
    people = {}
    years = {}
    for job_index, side, role, ready in first_meta:
        if not ready:
            continue
        text = raw_first[cursor]
        cursor += 1
        if role == "director":
            people[(job_index, side)] = parse_person(text)
        else:
            years[(job_index, side)] = parse_year(text)
    second_prompts = []
    second_meta = []
    for job_index, job in enumerate(jobs):
        spec = job["spec"]
        if not spec["kind"].startswith("director"):
            continue
        year_kind = "born" if spec["kind"].endswith("born") else "died"
        for side in ("left", "right"):
            person = people.get((job_index, side), "")
            prompt = hop_prompt(job["question"], person, year_kind, job["pages"]) if person else None
            second_prompts.append(prompt or "")
            second_meta.append((job_index, side, prompt is not None))
    raw_second = generator.generate([prompt for prompt in second_prompts if prompt])
    cursor = 0
    for job_index, side, ready in second_meta:
        if not ready:
            continue
        years[(job_index, side)] = parse_year(raw_second[cursor])
        cursor += 1

    totals = defaultdict(lambda: [0, 0, 0, 0])
    overall = defaultdict(lambda: [0, 0, 0])
    shown = defaultdict(int)
    for key, row in stored.items():
        overall[key[0]][0] += 1
        overall[key[0]][1] += exact_match(canonicalize_polar(row["question"], row["answer"]), row["gold"])
    for job_index, job in enumerate(jobs):
        spec = job["spec"]
        left_year = years.get((job_index, "left"))
        right_year = years.get((job_index, "right"))
        structural = decide(spec["left"], spec["right"], left_year, right_year, spec["later"])
        cover = canonicalize_polar(job["question"], job["cover"])
        pred = structural or cover
        bucket = totals[job["dataset"]]
        bucket[0] += 1
        bucket[1] += exact_match(cover, job["gold"])
        bucket[2] += exact_match(pred, job["gold"])
        if structural:
            bucket[3] += 1
            totals[job["dataset"] + ":fire"][0] += 1
            totals[job["dataset"] + ":fire"][1] += exact_match(structural, job["gold"])
            if not exact_match(structural, job["gold"]) and shown[job["dataset"]] < 5:
                shown[job["dataset"]] += 1
                print(
                    "MISS",
                    job["dataset"],
                    job["condition"],
                    "gold=",
                    job["gold"],
                    "pred=",
                    structural,
                    "years",
                    left_year,
                    right_year,
                    flush=True,
                )
                print(" ", job["question"][:160], flush=True)
        gain = exact_match(pred, job["gold"]) - exact_match(cover, job["gold"])
        overall[job["dataset"]][2] += gain
    for dataset, (n, cover_hit, hybrid_hit, fired) in totals.items():
        if dataset.endswith(":fire"):
            continue
        fire = totals[dataset + ":fire"]
        precision = 100 * fire[1] / fire[0] if fire[0] else 0.0
        all_n, all_cover, extra = overall[dataset]
        print(
            f"{dataset} comparative n {n} cover {100 * cover_hit / n:.2f} "
            f"hybrid {100 * hybrid_hit / n:.2f} fire {100 * fired / n:.1f} precision {precision:.1f}",
            flush=True,
        )
        print(
            f"{dataset} stress n {all_n} cover {100 * all_cover / all_n:.2f} "
            f"hybrid {100 * (all_cover + extra) / all_n:.2f}",
            flush=True,
        )
    del generator
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
