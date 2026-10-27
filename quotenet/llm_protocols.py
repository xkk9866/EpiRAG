"""Prompted LLM agents on exactly the views QuoteNet sees.

Belief-space protocols (what crosses a view boundary is an answer):
  vote     each agent answers from its view; majority of non-refusals
  debate   Du et al. (2024): agents see the others' answers and revise once
  madam    MADAM-RAG (Wang et al., 2025): answers with explanations, an
           aggregator summarizes, up to three rounds
Evidence-space protocol (the principle of QuoteNet, zero-shot):
  quote    each agent copies at most K words of its own documents for the
           others, without answering; every agent then answers from its view
           and the peers' quotes; majority of non-refusals
Decomposition (Self-Ask, Press et al. 2023; the private-view analogue of
IRCoT-style interleaving):
  selfask  a coordinator without documents asks follow-up questions; every
           agent answers each one from its own view; the majority
           intermediate answer is appended; at most four follow-ups
Reference:
  pool     one reader holds every page (no privacy)

All agents use one answer prompt; methods differ only in what crosses the
boundary. Exchanged tokens are counted with the LLM's tokenizer.

    python -m quotenet.llm_protocols --llm hf:Qwen/Qwen2.5-7B-Instruct@int8
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from epinet.data.examples import sample_examples
from epinet.llm.backend import make_llm
from epinet.text import best_em, best_f1, canonical_polar, is_polar, is_refusal, normalize_answer
from quotenet.views import REGIMES, build_views, union


def render(pages) -> str:
    return "\n".join(f"[{i}] {p.title}\n{p.text}" for i, p in enumerate(pages, 1)) or "(no documents)"


def answer_prompt(q, pages, notes=None, note_head="Notes from other agents"):
    polar = "If it is a yes/no question, answer yes or no.\n" if is_polar(q) else ""
    extra = ""
    if notes:
        extra = f"\n{note_head}:\n" + "\n".join(f"- {n}" for n in notes)
    return ("Answer the question using your documents"
            + (" and the notes" if notes else "")
            + ". If they do not contain the answer, write unknown.\n" + polar
            + "Reply with the short answer only. Do not explain.\n\n"
            + f"Question: {q}\nDocuments:\n{render(pages)}{extra}")


def quote_prompt(q, pages, k):
    return ("You hold private documents. Other agents hold different documents and must answer the question. "
            f"Copy, word for word, the part of your documents that the other agents most need (at most {k} words). "
            "Do not answer the question and do not add words of your own.\n\n"
            f"Question: {q}\nDocuments:\n{render(pages)}")


def madam_agent(q, pages, summary):
    extra = f"\nSummary of the previous discussion: {summary}" if summary else ""
    return ("Answer the question using only your documents. Some documents may be wrong or irrelevant. "
            "Reply in two lines:\nAnswer: <short answer>\nExplanation: <one sentence>\n\n"
            f"Question: {q}\nDocuments:\n{render(pages)}{extra}")


def madam_aggregate(q, replies):
    body = "\n".join(f"Agent {i + 1}: {r}" for i, r in enumerate(replies))
    return ("Agents answered from different documents; some may be misinformed or lack evidence. Weigh their "
            "explanations, discard unsupported answers, and give the final answer.\n"
            "Reply in two lines:\nAnswer: <short answer>\nSummary: <one sentence>\n\n"
            f"Question: {q}\n{body}")


SELFASK = """Answer the question by asking follow-up questions. Other agents hold the documents and answer each follow-up question; you only see their answers. Ask one follow-up question at a time. When you know the answer, write the final answer.

Question: Who was the father of the director of Blind Shaft?
Are follow up questions needed here: Yes.
Follow up: Who directed Blind Shaft?
Intermediate answer: Li Yang.
Follow up: Who was the father of Li Yang?
Intermediate answer: Li Wenhua.
So the final answer is: Li Wenhua

Question: Which film came out first, Jaws or Alien?
Are follow up questions needed here: Yes.
Follow up: When did Jaws come out?
Intermediate answer: 1975.
Follow up: When did Alien come out?
Intermediate answer: 1979.
So the final answer is: Jaws

Question: Are Vogue and Elle both fashion magazines?
Are follow up questions needed here: Yes.
Follow up: What kind of publication is Vogue?
Intermediate answer: a fashion magazine.
Follow up: What kind of publication is Elle?
Intermediate answer: a fashion magazine.
So the final answer is: yes

Question: {question}
Are follow up questions needed here: Yes.
{trace}"""


def selfask_step(text: str):
    for line in (text or "").splitlines():
        line = line.strip()
        low = line.lower()
        if low.startswith("so the final answer is"):
            return "final", line.split(":", 1)[-1].strip().rstrip(".")
        if low.startswith("follow up:"):
            return "ask", line.split(":", 1)[1].strip()
    return "final", short(text)


def short(text: str) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    if line.lower().startswith("answer:"):
        line = line[7:]
    return line.strip().strip("*").strip().rstrip(".")


def field(text: str, name: str) -> str:
    for line in (text or "").splitlines():
        if line.strip().lower().startswith(name.lower() + ":"):
            return line.split(":", 1)[1].strip().strip("*").strip().rstrip(".")
    return short(text)


def majority(answers: list[str]) -> str:
    keep = [a for a in answers if not is_refusal(a)]
    if not keep:
        return answers[0] if answers else ""
    top = Counter(normalize_answer(a) for a in keep).most_common(1)[0][0]
    return next(a for a in keep if normalize_answer(a) == top)


def clip_words(text: str, k: int) -> str:
    return " ".join((text or "").replace("\n", " ").split()[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="hf:Qwen/Qwen2.5-7B-Instruct@int8")
    ap.add_argument("--datasets", default="hotpotqa,2wiki,musique")
    ap.add_argument("--n-dev", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--k", type=int, default=32, help="words per quote")
    ap.add_argument("--methods", default="vote,debate,quote,madam,pool")
    ap.add_argument("--out", default="experiments/quote_llm")
    ap.add_argument("--tag", default="", help="suffix of the output file")
    a = ap.parse_args()
    llm = make_llm(a.llm)
    methods = a.methods.split(",")
    os.makedirs(a.out, exist_ok=True)
    tag = a.llm.split("/")[-1].replace(":", "_")
    eps = []
    for d in a.datasets.split(","):
        for ex in sample_examples(d, "validation", a.n_test, seed=7, offset=a.n_dev):  # QuoteNet's test block
            for r in REGIMES:
                eps.append((ex, r, build_views(ex, r)))
    q = lambda e: e[0].question
    pol = lambda e, s: canonical_polar(q(e), short(s))
    A = 3
    preds = {m: [""] * len(eps) for m in methods}
    cost = {m: [0] * len(eps) for m in methods}

    local = llm.generate([answer_prompt(q(e), v) for e in eps for v in e[2]], 24)
    local = [[pol(e, local[A * i + j]) for j in range(A)] for i, e in enumerate(eps)]
    if "vote" in methods:
        preds["vote"] = [majority(x) for x in local]
    if "debate" in methods:
        deb = llm.generate([answer_prompt(q(e), v, [local[i][k] for k in range(A) if k != j], "Answers of other agents")
                            for i, e in enumerate(eps) for j, v in enumerate(e[2])], 24)
        for i, e in enumerate(eps):
            preds["debate"][i] = majority([pol(e, deb[A * i + j]) for j in range(A)])
            cost["debate"][i] = sum((A - 1) * llm.count(x) for x in local[i])
    if "quote" in methods:
        quotes = llm.generate([quote_prompt(q(e), v, a.k) for e in eps for v in e[2]], 2 * a.k)
        quotes = [[clip_words(quotes[A * i + j], a.k) for j in range(A)] for i in range(len(eps))]
        ans = llm.generate([answer_prompt(q(e), v, [quotes[i][k] for k in range(A) if k != j])
                            for i, e in enumerate(eps) for j, v in enumerate(e[2])], 24)
        for i, e in enumerate(eps):
            preds["quote"][i] = majority([pol(e, ans[A * i + j]) for j in range(A)])
            cost["quote"][i] = sum((A - 1) * llm.count(x) for x in quotes[i])
    if "pool" in methods:
        pool = llm.generate([answer_prompt(q(e), union(e[2], e[0], e[1])) for e in eps], 24)
        preds["pool"] = [pol(e, s) for e, s in zip(eps, pool)]
    if "selfask" in methods:
        trace, active = [""] * len(eps), list(range(len(eps)))
        for _hop in range(5):
            steps = llm.generate([SELFASK.format(question=q(eps[i]), trace=trace[i]) for i in active], 48)
            asks = []
            for i, out in zip(active, steps):
                kind, val = selfask_step(out)
                if kind == "final" or _hop == 4:
                    preds["selfask"][i] = canonical_polar(q(eps[i]), val)
                else:
                    asks.append((i, val))
            if not asks:
                break
            sub = llm.generate([answer_prompt(f, v) for i, f in asks for v in eps[i][2]], 24)
            active = []
            for k, (i, f) in enumerate(asks):
                inter = majority([short(x) for x in sub[A * k:A * k + A]])
                trace[i] += f"Follow up: {f}\nIntermediate answer: {inter}.\n"
                # the follow-up goes to every agent; every agent's answer goes back to the coordinator
                cost["selfask"][i] += A * llm.count(f) + sum(llm.count(short(x)) for x in sub[A * k:A * k + A])
                active.append(i)
    if "madam" in methods:
        summaries, finals, active = [""] * len(eps), [""] * len(eps), list(range(len(eps)))
        for _round in range(3):
            outs = llm.generate([madam_agent(q(eps[i]), v, summaries[i]) for i in active for v in eps[i][2]], 80)
            aggs = llm.generate([madam_aggregate(q(eps[i]), outs[A * k:A * k + A]) for k, i in enumerate(active)], 80)
            nxt = []
            for k, i in enumerate(active):
                ans_i = canonical_polar(q(eps[i]), field(aggs[k], "Answer"))
                summ = aggs[k].split("Summary:", 1)[-1].strip() if "Summary:" in aggs[k] else ""
                cost["madam"][i] += sum(llm.count(x) for x in outs[A * k:A * k + A]) + A * llm.count(summ)
                if normalize_answer(ans_i) != normalize_answer(finals[i]):
                    nxt.append(i)
                finals[i], summaries[i] = ans_i, summ
            active = nxt
            if not active:
                break
        preds["madam"] = finals
    path = os.path.join(a.out, f"{tag}_k{a.k}{a.tag}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for i, (ex, r, _v) in enumerate(eps):
            for m in methods:
                ans = preds[m][i]
                fh.write(json.dumps({"qid": ex.qid, "dataset": ex.dataset, "qtype": ex.qtype, "regime": r,
                                     "method": m, "answer": ans, "gold": ex.answers, "tokens": cost[m][i],
                                     "em": best_em(ans, ex.answers), "f1": best_f1(ans, ex.answers)},
                                    ensure_ascii=False) + "\n")
    print("done", path, "llm calls", llm.calls)


if __name__ == "__main__":
    main()
