"""Recompute belief states for the smoke examples with the fixed hypothesis."""

import json
from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import NLI, coordinate
from epirag.run_eval import materialize

rows = [json.loads(line) for line in open("experiments/results/smoke.jsonl", encoding="utf-8")]
local = {}
for row in rows:
    if row["method"] == "epirag":
        local[(row["dataset"], row["qid"], row["condition"])] = row["local_answers"]

hot = load_dataset("hotpot_qa", "distractor", split="validation")
wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
_, hot_eval = select_examples(hot, "hotpotqa", 1, 12, 7)
_, wiki_eval = select_examples(wiki, "2wiki", 1, 12, 7)
instances = [
    build_instance(hot[hot_eval[0]], "hotpotqa"),
    build_instance(wiki[wiki_eval[0]], "2wiki"),
]
nli = NLI("cross-encoder/nli-deberta-v3-base", "cuda:1")
for inst in instances:
    for condition in ("clean", "asymmetry", "corruption", "temporal", "incomplete"):
        answers = local[(inst.dataset, inst.qid, condition)]
        agents, archive = materialize(inst, condition)
        state = coordinate(inst.question, archive, agents, answers, nli)
        print(
            f"{inst.dataset:8} {condition:12} trig={state['triggered']} "
            f"margin={state['margin']:.3f} hat={state['hat'][:40]!r} "
            f"beliefs={[round(b, 3) for b in state['beliefs']]} "
            f"n_bundle={len(state['bundle'])} claims={state['claims']}"
        )
