"""Is the gold span even in the chain candidate set? Development only."""

from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.diag_chainsearch import STRESS, chain_scored, pages_of
from epirag.metrics import exact_match, normalize_answer

EXAMPLES = 6


def main():
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    specs = (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160))
    for name, data, budget in specs:
        _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block="advance")
        instances = [build_instance(data[i], name) for i in ids]
        n = in_set = top1 = 0
        sizes = []
        shown = 0
        by_type = defaultdict(lambda: [0, 0])
        for instance in instances:
            for condition in STRESS:
                scored = chain_scored(instance.question, pages_of(instance, condition))
                n += 1
                sizes.append(len(scored))
                gold = normalize_answer(instance.answer)
                keys = list(scored)
                hit = any(exact_match(key, instance.answer) for key in keys)
                in_set += int(hit)
                by_type[instance.qtype][0] += int(hit)
                by_type[instance.qtype][1] += 1
                if keys:
                    best = max(scored.items(), key=lambda item: item[1][0])[0]
                    top1 += exact_match(best, instance.answer)
                    if shown < EXAMPLES and condition == "asymmetry" and not exact_match(best, instance.answer):
                        shown += 1
                        print("Q", instance.question[:140])
                        print(" gold", instance.answer, "| top", best, "| set", keys[:8])
        print(
            f"{name} n {n} gold-in-set {100 * in_set / n:.1f} top1 {100 * top1 / n:.1f} "
            f"mean-set {sum(sizes) / n:.1f}",
            flush=True,
        )
        for qtype, (hit, total) in sorted(by_type.items()):
            print(f"  {qtype} {100 * hit / total:.1f}", flush=True)


if __name__ == "__main__":
    main()
