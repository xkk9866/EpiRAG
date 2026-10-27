"""Print each derivation step on the advance smoke items. No scores, no folio."""

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.derive import (
    _acceptable,
    _focus,
    _stated,
    parse_plan,
    plan_prompt,
    resolve_derivations,
)
from epirag.engine import Generator, evidence_prompt, parse_answer
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
    generator = Generator(
        "Qwen/Qwen2.5-7B-Instruct",
        "cuda:1",
        "experiments/cache/generations.jsonl",
        batch_size=2,
    )
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    items = []
    for name, data, budget in (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160)):
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="advance")
        for index in ids[:4]:
            instance = build_instance(data[index], name)
            items.append((name, instance.qtype, instance.question, pages_of(instance)))
    raw_plans = generator.generate(
        [plan_prompt(question) for _name, _qtype, question, _pages in items],
        max_new_tokens=96,
    )
    for (name, qtype, question, pages), raw in zip(items, raw_plans):
        spec = parse_plan(raw)
        print(f"\n== {name} {qtype}", flush=True)
        if spec is None:
            print("unparsed", flush=True)
            continue
        if spec["kind"] == "chain":
            questions = [spec["question1"]]
            if spec["question2"]:
                questions.append(spec["question2"])
        else:
            questions = [spec["question1"], spec["question2"]]
        bridge = ""
        for step, asked in enumerate(questions, start=1):
            query = asked.replace("PERSON", bridge) if bridge else asked
            evidence = _focus(query, pages)
            prompt = evidence_prompt(query, evidence) if evidence else ""
            raw_step = generator.generate([prompt])[0] if prompt else ""
            answer, _quote = parse_answer(raw_step)
            answer = " ".join(answer.split())
            print(
                f"step {step} ok={_acceptable(answer) and _stated(answer, evidence)} "
                f"answer={answer!r} q={query}",
                flush=True,
            )
            bridge = answer
    finals = resolve_derivations(
        generator.generate,
        [(question, pages) for _name, _qtype, question, pages in items],
        generator.count,
    )
    for (name, qtype, _question, _pages), final in zip(items, finals):
        print(f"{name} {qtype} -> {final['decision']} {final['answer']!r}", flush=True)


if __name__ == "__main__":
    main()
