"""Format smoke for the derivation planner.

Uses the already-scored advance block only to see whether the reader emits a
parseable plan. It does not write a result file and it does not score folio.
"""

import sys

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.derive import parse_plan, plan_prompt, resolve_derivations
from epirag.engine import Generator
from epirag.run_eval import materialize


def pages_of(instance):
    agents, _archive = materialize(instance, "clean")
    pages = []
    seen = set()
    for group in agents:
        for passage in group:
            if passage.pid in seen:
                continue
            seen.add(passage.pid)
            pages.append(passage)
    return pages


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    print("loading generator", flush=True)
    generator = Generator(
        "Qwen/Qwen2.5-7B-Instruct",
        device,
        "experiments/cache/generations.jsonl",
        batch_size=2,
    )
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    for name, data, budget in (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160)):
        _dev, folio_ids = select_examples(data, name, budget, n_dev=12, seed=7, block="folio")
        print(f"{name} folio {len(folio_ids)} budget {budget}", flush=True)
    items = []
    labels = []
    for name, data, budget in (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160)):
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="advance")
        for index in ids[:4]:
            instance = build_instance(data[index], name)
            items.append((instance.question, pages_of(instance)))
            labels.append(f"{name} {instance.qtype}")
    raw_plans = generator.generate([plan_prompt(question) for question, _pages in items], max_new_tokens=96)
    parsed = 0
    for label, (question, _pages), raw in zip(labels, items, raw_plans):
        spec = parse_plan(raw)
        parsed += spec is not None
        print(f"\n== {label}", flush=True)
        print(question, flush=True)
        print(raw, flush=True)
        print("PARSED" if spec else "UNPARSED", spec["kind"] if spec else "", flush=True)
    print(f"\nparsed {parsed}/{len(items)}", flush=True)
    finals = resolve_derivations(generator.generate, items, generator.count)
    for label, final in zip(labels, finals):
        print(f"{label} decision {final['decision']} answer {final['answer']}", flush=True)


if __name__ == "__main__":
    main()
