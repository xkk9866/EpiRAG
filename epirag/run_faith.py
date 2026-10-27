"""FaithEval inconsistent contexts, with the slice locked before scoring.

Seed 7 takes the first 400 rows of a shuffle over the 1,500 test items.
Each item has two documents. They are assigned to two private views.
The third view is empty. Every method receives the same instruction:
if the evidence supports two different answers, reply conflict.
Extra retrieval is off because this benchmark has no archive.
"""

from __future__ import annotations

import json
import os
import random

from datasets import load_dataset

from epirag.data import Passage
from epirag.engine import (
    Generator,
    NLI,
    coordinate,
    debate_prompt,
    evidence_prompt,
    parse_answer,
)
from epirag.metrics import normalize_answer
from epirag.run_eval import majority

INSTRUCTION = "If the evidence supports two different answers, reply conflict."
N_EVAL = 400
SEED = 7
OUT = "experiments/results/records_faith.jsonl"
METHODS = (
    "centralized",
    "debate",
    "epirag_cover",
    "epirag_flat",
    "epirag_indep",
    "epirag_graph",
    "epirag_bridge",
    "epirag_gate",
)


def conflict_hit(answer: str) -> float:
    return float(normalize_answer(answer) == "conflict")


def loose_hit(answer: str) -> float:
    text = normalize_answer(answer)
    phrases = (
        "conflict",
        "multiple answers",
        "disagreement",
        "inconsistent",
        "contradictory",
        "contradiction",
        "two answers",
    )
    return float(any(phrase in text for phrase in phrases))


def side_hit(answer: str, golds: list[str]) -> float:
    pred = normalize_answer(answer)
    return float(any(pred == normalize_answer(gold) for gold in golds))


def split_documents(context: str) -> list[str]:
    parts = [part.strip() for part in context.split("Document:") if part.strip()]
    if len(parts) < 2:
        raise ValueError("expected two documents")
    return parts[:2]


def load_done(path: str) -> set[tuple]:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["qid"], row["method"]))
    return done


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    data = load_dataset("Salesforce/FaithEval-inconsistent-v1.0", split="test")
    order = list(range(len(data)))
    random.Random(SEED).shuffle(order)
    chosen = [data[index] for index in order[:N_EVAL]]
    done = load_done(OUT)
    pending = [
        row
        for row in chosen
        if any((row["qid"], method) not in done for method in METHODS)
    ]
    print(f"pending {len(pending)} of {N_EVAL}", flush=True)
    nli = NLI("cross-encoder/nli-deberta-v3-base", "cuda:1")
    generator = Generator(
        "Qwen/Qwen2.5-7B-Instruct",
        "cuda:1",
        "experiments/cache/generations.jsonl",
        batch_size=4,
    )
    chunk_size = 8
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        local_prompts = []
        prepared = []
        for row in chunk:
            docs = split_documents(row["context"])
            passages = [
                Passage(f"d{i}", f"Document {i + 1}", text, False, 2020, f"s{i}")
                for i, text in enumerate(docs)
            ]
            empty = Passage(
                "d2",
                "Unassigned",
                "No page was assigned to this view.",
                False,
                2020,
                "s2",
            )
            agents = [[passages[0]], [passages[1]], [empty]]
            for group in agents:
                local_prompts.append(evidence_prompt(row["question"], group, INSTRUCTION))
            prepared.append((row, agents, passages))
        local_raw = generator.generate(local_prompts)
        cursor = 0
        second_prompts = []
        second_meta = []
        states = {}
        locals_by = {}
        for index, (row, agents, passages) in enumerate(prepared):
            local = []
            for _agent in agents:
                answer, _quote = parse_answer(local_raw[cursor])
                local.append(answer)
                cursor += 1
            locals_by[index] = local
            for mode, name, conflict_ok in (
                ("cover", "epirag_cover", False),
                ("flat", "epirag_flat", False),
                ("indep", "epirag_indep", True),
                ("graph", "epirag_graph", False),
                ("bridge", "epirag_bridge", True),
                ("gate", "epirag_gate", True),
            ):
                state = coordinate(
                    row["question"],
                    [],
                    agents,
                    local,
                    nli,
                    allow_retrieval=False,
                    mode=mode,
                    allow_conflict=conflict_ok,
                )
                states[(index, name)] = state
                if state.get("structural"):
                    continue
                bundle = []
                seen = set()
                for snippet in state["bundle"]:
                    if snippet.passage.pid in seen:
                        continue
                    seen.add(snippet.passage.pid)
                    bundle.append(snippet.passage)
                second_prompts.append(evidence_prompt(row["question"], bundle[:4], INSTRUCTION))
                second_meta.append((index, name))
            second_prompts.append(evidence_prompt(row["question"], passages, INSTRUCTION))
            second_meta.append((index, "centralized"))
            for agent, group in enumerate(agents):
                others = [local[j] for j in range(3) if j != agent]
                second_prompts.append(
                    debate_prompt(row["question"], group, others, INSTRUCTION)
                )
                second_meta.append((index, f"debate_{agent}"))
        second_raw = generator.generate(second_prompts)
        grouped = {}
        for meta, raw in zip(second_meta, second_raw):
            grouped.setdefault(meta[0], {})[meta[1]] = raw
        for index, (row, agents, passages) in enumerate(prepared):
            local = locals_by[index]
            raw_map = grouped[index]
            debate_answers = [
                parse_answer(raw_map[f"debate_{agent}"])[0] for agent in range(3)
            ]
            outputs = {
                "centralized": parse_answer(raw_map["centralized"])[0],
                "debate": majority(debate_answers),
            }
            for name in ("epirag_cover", "epirag_flat", "epirag_indep", "epirag_graph", "epirag_bridge", "epirag_gate"):
                structural = states[(index, name)].get("structural") or ""
                if structural:
                    outputs[name] = structural
                else:
                    outputs[name] = parse_answer(raw_map[name])[0]
            for method, answer in outputs.items():
                key = (row["qid"], method)
                if key in done:
                    continue
                record = {
                    "qid": row["qid"],
                    "question": row["question"],
                    "golds": list(row["answers"]),
                    "method": method,
                    "answer": answer,
                    "conflict": conflict_hit(answer),
                    "loose": loose_hit(answer),
                    "side": side_hit(answer, list(row["answers"])),
                    "local_answers": local,
                }
                if method.startswith("epirag"):
                    state = states[(index, method)]
                    record["margin"] = state["margin"]
                    record["hat"] = state["hat"]
                append_row(OUT, record)
                done.add(key)
        print(f"finished {min(start + chunk_size, len(pending))}/{len(pending)}", flush=True)
    summarize()


def summarize():
    if not os.path.exists(OUT):
        return
    buckets = {}
    with open(OUT, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            buckets.setdefault(row["method"], []).append(row)
    for method, rows in sorted(buckets.items()):
        n = len(rows)
        conflict = 100 * sum(row["conflict"] for row in rows) / n
        loose = 100 * sum(row["loose"] for row in rows) / n
        side = 100 * sum(row["side"] for row in rows) / n
        print(
            f"{method:16} n={n} conflict={conflict:.1f} "
            f"loose={loose:.1f} side={side:.1f}"
        )


if __name__ == "__main__":
    main()
