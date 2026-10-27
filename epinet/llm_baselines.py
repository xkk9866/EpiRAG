"""Prompted LLM multi-agent baselines on exactly the views EpiNet sees.

vote     each agent answers from its view; majority string
pool     one reader over the union of views (centralized RAG)
debate   Du et al. (2023): two rounds, agents see the others' answers; majority
madam    MADAM-RAG (Wang et al., 2025): agents answer with a short explanation,
         an aggregator summarizes; up to three rounds, stop when stable

    python -m epinet.llm_baselines --llm hf:Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from epinet.data.examples import sample_examples
from epinet.data.views import CONDITIONS, build_views
from epinet.llm.backend import make_llm
from epinet.llm.prompts import agent_prompt, parse_agent, parse_short, reader_prompt, render_pages
from epinet.text import best_em, best_f1, canonical_polar, normalize_answer


def majority(answers: list[str]) -> str:
    keys = [normalize_answer(a) for a in answers if normalize_answer(a)]
    if not keys:
        return ""
    top = Counter(keys).most_common(1)[0][0]
    return next(a for a in answers if normalize_answer(a) == top)


def debate_prompt(q, pages, others):
    return (reader_prompt(q, pages) + "\nOther agents answered: " + "; ".join(others)
            + "\nRevise your answer only if your evidence supports theirs. Reply with the short answer span only.")


def madam_agent(q, pages, summary):
    extra = f"\nSummary of the previous discussion: {summary}" if summary else ""
    return (
        "Answer the question using only your document. Some documents may be wrong or outdated. "
        "Reply in two lines:\nAnswer: <short answer>\nExplanation: <one sentence>\n\n"
        f"Question: {q}\nDocument:\n{render_pages(pages)}{extra}"
    )


def madam_aggregate(q, replies):
    body = "\n".join(f"Agent {i + 1}: {r}" for i, r in enumerate(replies))
    return (
        "Agents answered from different documents; some may be misinformed. Weigh their explanations, "
        "discard unsupported or conflicting misinformation, and give the final answer.\n"
        "Reply in two lines:\nAnswer: <short answer>\nSummary: <one sentence>\n\n"
        f"Question: {q}\n{body}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="hf:Qwen/Qwen2.5-7B-Instruct@int8")
    ap.add_argument("--datasets", default="hotpotqa,2wiki,musique")
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--out", default="experiments/llm_baselines")
    a = ap.parse_args()
    llm = make_llm(a.llm)
    os.makedirs(a.out, exist_ok=True)
    tag = a.llm.split("/")[-1].replace(":", "_")
    path = os.path.join(a.out, f"{tag}.jsonl")
    eps = []
    for d in a.datasets.split(","):
        for ex in sample_examples(d, "validation", a.n_test, seed=7, offset=a.n_dev):  # same test block as EpiNet
            for c in CONDITIONS:
                eps.append((ex, c, build_views(ex, c)))
    q = lambda e: e[0].question
    # round 0: local answers
    local = llm.generate([agent_prompt(q(e), v) for e in eps for v in e[2]], 80)
    local = [[canonical_polar(q(e), parse_agent(local[3 * i + j])[0]) for j in range(3)] for i, e in enumerate(eps)]
    pool = llm.generate([reader_prompt(q(e), [p for v in e[2] for p in v]) for e in eps], 32)
    deb = llm.generate([debate_prompt(q(e), v, [local[i][k] for k in range(3) if k != j])
                        for i, e in enumerate(eps) for j, v in enumerate(e[2])], 32)
    # MADAM-RAG rounds
    summaries = [""] * len(eps)
    finals = [""] * len(eps)
    active = list(range(len(eps)))
    for _round in range(3):
        outs = llm.generate([madam_agent(q(eps[i]), v, summaries[i]) for i in active for v in eps[i][2]], 80)
        aggs = llm.generate([madam_aggregate(q(eps[i]), outs[3 * k:3 * k + 3]) for k, i in enumerate(active)], 80)
        nxt = []
        for k, i in enumerate(active):
            ans, _ = parse_agent(aggs[k])
            ans = canonical_polar(q(eps[i]), ans)
            summ = aggs[k].split("Summary:", 1)[-1].strip() if "Summary:" in aggs[k] else ""
            if normalize_answer(ans) != normalize_answer(finals[i]):
                nxt.append(i)
            finals[i], summaries[i] = ans, summ
        active = nxt
        if not active:
            break
    with open(path, "w", encoding="utf-8") as fh:
        for i, (ex, c, _v) in enumerate(eps):
            preds = {
                "vote": majority(local[i]),
                "pool": canonical_polar(ex.question, parse_short(pool[i])),
                "debate": majority([canonical_polar(ex.question, parse_short(deb[3 * i + j])) for j in range(3)]),
                "madam": finals[i],
            }
            for m, ans in preds.items():
                fh.write(json.dumps({"qid": ex.qid, "dataset": ex.dataset, "qtype": ex.qtype, "condition": c,
                                     "method": m, "answer": ans, "gold": ex.answers,
                                     "em": best_em(ans, ex.answers), "f1": best_f1(ans, ex.answers)},
                                    ensure_ascii=False) + "\n")
    print("done", path, "llm calls", llm.calls)


if __name__ == "__main__":
    main()
