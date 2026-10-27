"""Development: replace a reader string by the evidence span it overlaps."""

import json
import re
from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match, token_f1
from epirag.run_eval import materialize

STRESS = {"asymmetry", "corruption", "temporal", "incomplete"}
FILES = {
    "advance": ("experiments/results/records_advance.jsonl", "advance"),
    "forward": ("experiments/results/records_forward.jsonl", "forward"),
}


def spans(text):
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or "")
    found = []
    for size in range(1, 7):
        for i in range(0, len(words) - size + 1):
            found.append(" ".join(words[i : i + size]))
    return found


def project(answer, agents):
    best = answer
    best_f1 = 0.5
    for group in agents:
        for passage in group:
            for span in spans(passage.text):
                score = token_f1(answer, span)
                if score > best_f1 or (score == best_f1 and len(span) > len(best)):
                    best_f1 = score
                    best = span
    return best


def canon(question, answer):
    if answer == "conflict":
        return answer
    return canonicalize_polar(question, answer)


def main():
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    for label, (path, block) in FILES.items():
        stored = defaultdict(dict)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["condition"] not in STRESS:
                    continue
                if row["method"] not in ("centralized", "epirag_cover"):
                    continue
                stored[(row["dataset"], row["qid"], row["condition"])][row["method"]] = row
        instances = {}
        budgets = {"hotpotqa": 200, "2wiki": 160}
        for name, data in (("hotpotqa", hotpot), ("2wiki", wiki)):
            _dev, ids = select_examples(data, name, budgets[name], n_dev=12, seed=7, block=block)
            for index in ids:
                inst = build_instance(data[index], name)
                instances[(name, inst.qid)] = inst
        print("\n", label, flush=True)
        for dataset in ("hotpotqa", "2wiki"):
            for method in ("centralized", "epirag_cover"):
                n = raw_hit = proj_hit = 0
                for key, methods in stored.items():
                    if key[0] != dataset:
                        continue
                    inst = instances[(dataset, key[1])]
                    agents, _archive = materialize(inst, key[2])
                    answer = canon(inst.question, methods[method]["answer"])
                    projected = canon(inst.question, project(answer, agents))
                    n += 1
                    raw_hit += exact_match(answer, inst.answer)
                    proj_hit += exact_match(projected, inst.answer)
                print(
                    dataset,
                    method,
                    "raw", round(100 * raw_hit / n, 2),
                    "project", round(100 * proj_hit / n, 2),
                    flush=True,
                )


if __name__ == "__main__":
    main()
