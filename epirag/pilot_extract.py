"""Development only: rank evidence spans by frozen NLI support.

Not a reported test. The advance block already exists, so this only asks
whether an extractive ranker beats the readers before any new slice is spent.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import torch
from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import NLI, _stated_in, canonicalize_polar, claim_hypothesis, split_sentences
from epirag.metrics import exact_match, normalize_answer
from epirag.run_eval import materialize

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
STOP = {
    "the", "and", "for", "with", "from", "that", "this", "what", "which", "who",
    "when", "where", "how", "are", "was", "were", "has", "have", "had", "his",
    "her", "their", "its", "not", "but", "also", "into", "over", "after", "before",
}


def holding_sentence(span: str, agents) -> str:
    for group in agents:
        for passage in group:
            for sentence in split_sentences(passage.text) or [passage.text]:
                if _stated_in(sentence, span):
                    return sentence
    return ""


def candidates(question: str, agents, local_answers: list[str]) -> list[tuple[str, str]]:
    del question
    found: list[tuple[str, str]] = []
    seen = set()

    def add(span: str, sentence: str):
        text = re.sub(r"\s+", " ", (span or "").strip(" ,;:"))
        key = normalize_answer(text)
        if not key or key in seen or key in STOP or len(key) < 2:
            return
        if not sentence:
            sentence = holding_sentence(text, agents)
        if not sentence:
            return
        seen.add(key)
        found.append((text, sentence))

    for answer in local_answers:
        add(answer, "")
    for group in agents:
        for passage in group:
            add(passage.title, passage.text)
            for sentence in split_sentences(passage.text) or [passage.text]:
                for match in re.findall(
                    r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,4}\b",
                    sentence,
                ):
                    add(match, sentence)
                for match in re.findall(r"\b(?:1|2)\d{3}\b", sentence):
                    add(match, sentence)
                for word in ("yes", "no"):
                    if re.search(rf"\b{word}\b", sentence, flags=re.I):
                        add(word, sentence)
    return found[:48]


def main():
    stored = defaultdict(dict)
    with open("experiments/results/records_advance.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in STRESS or row["method"] != "centralized":
                continue
            stored[(row["dataset"], row["qid"], row["condition"])] = row
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    instances = {}
    for name, data, budget in (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160)):
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="advance")
        for index in ids:
            inst = build_instance(data[index], name)
            instances[(name, inst.qid)] = inst
    print("loading NLI", flush=True)
    nli = NLI("cross-encoder/nli-deberta-v3-base", "cuda:0")
    totals = defaultdict(lambda: defaultdict(int))
    prepared = []
    pairs = []
    for key, row in stored.items():
        dataset, qid, condition = key
        inst = instances[(dataset, qid)]
        agents, _archive = materialize(inst, condition)
        spans = candidates(inst.question, agents, row["local_answers"])
        start = len(pairs)
        for span, sentence in spans:
            pairs.append((sentence[:500], claim_hypothesis(inst.question, span)))
        prepared.append((dataset, inst, spans, start))
    print(f"questions {len(prepared)} pairs {len(pairs)}", flush=True)
    scored = nli.score(pairs, batch_size=64)
    for dataset, inst, spans, start in prepared:
        best_span = ""
        best_margin = -2.0
        for offset, (span, _sentence) in enumerate(spans):
            contra, entail, _neutral = scored[start + offset]
            margin = entail - contra
            if margin > best_margin:
                best_margin = margin
                best_span = span
        chosen = canonicalize_polar(inst.question, best_span)
        bucket = totals[dataset]
        bucket["n"] += 1
        bucket["hit"] += exact_match(chosen, inst.answer)
        if bucket["n"] % 200 == 0:
            print(dataset, bucket["n"], bucket["hit"], flush=True)
    for dataset, bucket in totals.items():
        n = bucket["n"]
        print(dataset, "extract", round(100 * bucket["hit"] / n, 2), "n", n, flush=True)
    del nli
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
