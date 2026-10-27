"""Run controlled EpiRAG experiments and write raw records."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from epirag.data import (
    CONDITIONS,
    Instance,
    Passage,
    build_instance,
    select_examples,
    temporal_passage_copy,
    tokenize,
)
from epirag.engine import (
    Generator,
    NLI,
    commit_answer,
    coordinate,
    debate_prompt,
    evidence_prompt,
    bridge_entity_prompt,
    chain_answer_prompt,
    marked_prompt,
    canonicalize_polar,
    MAX_CLAIM_TOKENS,
    parse_answer,
    revision_prompt,
    support_confidence,
)
from epirag.execute import executed_answers
from epirag.hop import resolve_hops
from epirag.derive import resolve_derivations
from epirag.metrics import exact_match, token_f1


MAIN_METHODS = (
    "independent",
    "centralized",
    "debate",
    "collab",
    "epirag",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Eval examples per dataset. 0 uses the full budget.")
    parser.add_argument("--hotpot", type=int, default=100)
    parser.add_argument("--wiki", type=int, default=80)
    parser.add_argument("--dev", type=int, default=0, help="If >0, evaluate the dev slice instead.")
    parser.add_argument(
        "--slice",
        choices=("eval", "confirm", "dev", "locked", "final", "holdout", "advance", "forward", "prospect", "ledger", "folio"),
        default="eval",
        help="eval is the slice used to compare rules. confirm is the next untouched block.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--gen-batch", type=int, default=4)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--nli", type=str, default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--out", type=str, default="experiments/results/records.jsonl")
    parser.add_argument("--cache", type=str, default="experiments/cache/generations.jsonl")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated methods to run. Existing methods in the output file are left unchanged.",
    )
    return parser.parse_args()


def order_passages(question: str, passages):
    if len(passages) <= 1:
        return list(passages)
    corpus = [tokenize(p.title + " " + p.text) or ["empty"] for p in passages]
    ranker = BM25Okapi(corpus)
    scores = ranker.get_scores(tokenize(question) or ["empty"])
    order = sorted(range(len(passages)), key=lambda i: float(scores[i]), reverse=True)
    return [passages[i] for i in order]


def materialize(instance: Instance, condition: str):
    agents = []
    for agent in range(3):
        passages = instance.view_passages(condition, agent)
        if condition == "temporal":
            passages = [temporal_passage_copy(p) for p in passages]
        passages = order_passages(instance.question, passages)
        agents.append(passages)
    archive = [p for p in instance.passages if str(p.pid).startswith("p")]
    if condition == "temporal":
        archive = [temporal_passage_copy(p) for p in archive]
    return agents, archive


def with_missing_gold(instance: Instance, agents, condition: str):
    """Oracle retrieval: one missing gold page, labeled as an external source."""
    present = {passage.pid for group in agents for passage in group}
    missing = [passage for passage in instance.passages if passage.gold and passage.pid not in present]
    if not missing:
        return agents
    page = missing[0]
    if condition == "temporal":
        page = temporal_passage_copy(page)
    injected = Passage(
        pid=page.pid,
        title=page.title,
        text=page.text,
        gold=True,
        year=page.year,
        source_id="oracle",
    )
    return [list(agents[0]) + [injected], list(agents[1]), list(agents[2])]


def pool_pages(question: str, agents, union):
    if union:
        return list(union)
    pages = []
    seen = set()
    for group in agents:
        for passage in group:
            if passage.pid in seen:
                continue
            seen.add(passage.pid)
            pages.append(passage)
    return order_passages(question, pages)[:6]


def majority(answers: list[str]) -> str:
    keyed = [(re_key(a), a) for a in answers if a]
    if not keyed:
        return ""
    counts = Counter(key for key, _raw in keyed)
    best_count = max(counts.values())
    winners = {key for key, count in counts.items() if count == best_count}
    for key, raw in keyed:
        if key in winners:
            return raw
    return keyed[0][1]


def re_key(text: str) -> str:
    return " ".join((text or "").lower().split())


def load_done(path: str) -> set[tuple]:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            done.add((row["dataset"], row["qid"], row["condition"], row["method"]))
    return done


def append_row(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_splits(args):
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    specs = []
    for name, data, budget in (("hotpotqa", hotpot, args.hotpot), ("2wiki", wiki, args.wiki)):
        block = "dev" if args.dev else args.slice
        dev_ids, eval_ids = select_examples(data, name, budget, n_dev=12, seed=args.seed, block=block)
        chosen = eval_ids
        if args.limit:
            chosen = chosen[: args.limit]
        instances = [build_instance(data[i], name) for i in chosen]
        type_counts = Counter(inst.qtype for inst in instances)
        print(f"{name}: {len(instances)} examples {dict(type_counts)}", flush=True)
        specs.append(instances)
    return [inst for group in specs for inst in group]


def run():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    # Load the reader before the datasets. Holding every passage and then
    # mapping the 7B weights has been crashing the process mid-load.
    print("loading generator", flush=True)
    generator = Generator(args.model, args.device, args.cache, batch_size=args.gen_batch)
    import gc
    gc.collect()
    print("loading NLI", flush=True)
    nli = NLI(args.nli, args.device)
    instances = build_splits(args)
    done = load_done(args.out)
    print(f"loaded {len(instances)} instances; {len(done)} existing records", flush=True)

    active_methods = list(MAIN_METHODS) + ["vote", "epirag_flat"]
    # reliability, contradiction filter, extra retrieval, coordination mode
    ablation_flags = {
        "epirag": (True, True, True, "typed"),
        "epirag_flat": (True, True, True, "flat"),
    }
    if not args.skip_ablations:
        ablation_flags.update(
            {
                "epirag_noreliability": (False, True, True, "typed"),
                "epirag_noretrieval": (True, True, False, "typed"),
                "epirag_nofilter": (True, False, True, "typed"),
            }
        )
        active_methods.extend(
            ["epirag_noreliability", "epirag_noretrieval", "epirag_nofilter"]
        )
    ablation_flags["epirag_cover"] = (True, True, True, "cover")
    ablation_flags["epirag_cover_noret"] = (True, True, False, "cover")
    ablation_flags["epirag_cover_norel"] = (False, True, True, "cover")
    active_methods.extend(["epirag_cover", "epirag_cover_noret", "epirag_cover_norel"])
    # Drops rival sentences without also shortening the claim text.
    ablation_flags["epirag_drop"] = (True, True, True, "drop")
    active_methods.append("epirag_drop")
    ablation_flags["epirag_indep"] = (True, True, True, "indep")
    ablation_flags["epirag_indep_always"] = (True, True, True, "indep_always")
    ablation_flags["epirag_indep_never"] = (True, True, False, "indep_never")
    ablation_flags["epirag_graph"] = (False, True, False, "graph")
    ablation_flags["epirag_bridge"] = (True, True, True, "bridge")
    ablation_flags["epirag_oracle"] = (True, True, False, "bridge")
    ablation_flags["epirag_gate"] = (False, True, True, "gate")
    ablation_flags["epirag_mark"] = (False, True, True, "mark")
    ablation_flags["epirag_chain"] = (True, True, True, "cover")
    ablation_flags["epirag_exec"] = (True, True, True, "cover")
    ablation_flags["epirag_hop"] = (True, True, True, "cover")
    ablation_flags["epirag_derive"] = (True, True, True, "cover")
    active_methods.extend(
        [
            "epirag_indep",
            "epirag_indep_always",
            "epirag_indep_never",
            "epirag_graph",
            "epirag_bridge",
            "epirag_oracle",
            "epirag_gate",
            "epirag_mark",
            "epirag_chain",
            "epirag_exec",
            "epirag_hop",
            "epirag_derive",
        ]
    )
    # Same witness bundle as cover. The reader may overturn a unanimous
    # committee only when the bundle supports the new span more strongly.
    ablation_flags["epirag_commit"] = (True, True, True, "cover")
    ablation_flags["epirag_commit_noret"] = (True, True, False, "cover")
    ablation_flags["epirag_commit_norel"] = (False, True, True, "cover")
    active_methods.extend(["epirag_commit", "epirag_commit_noret", "epirag_commit_norel"])
    gate_methods = {"epirag_commit", "epirag_commit_noret", "epirag_commit_norel"}
    # Cover bundle, but the reader revises proposals instead of answering in isolation.
    ablation_flags["epirag_revise"] = (True, True, True, "cover")
    ablation_flags["epirag_revise_noret"] = (True, True, False, "cover")
    ablation_flags["epirag_revise_norel"] = (False, True, True, "cover")
    active_methods.extend(["epirag_revise", "epirag_revise_noret", "epirag_revise_norel"])
    revise_methods = {"epirag_revise", "epirag_revise_noret", "epirag_revise_norel"}
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        ablation_flags = {name: flags for name, flags in ablation_flags.items() if name in wanted}
        active_methods = [name for name in active_methods if name in wanted]
        need_baselines = bool(wanted & {"independent", "centralized", "debate", "collab", "vote"})
    else:
        need_baselines = True
    pending = []
    for instance in instances:
        for condition in CONDITIONS:
            if all((instance.dataset, instance.qid, condition, method) in done for method in active_methods):
                continue
            pending.append((instance, condition))
    print(f"pending scenarios {len(pending)}", flush=True)

    chunk_size = 8
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        scenarios = []
        local_prompts = []
        for instance, condition in chunk:
            agents, archive = materialize(instance, condition)
            for passages in agents:
                local_prompts.append(evidence_prompt(instance.question, passages))
            scenarios.append((instance, condition, agents, archive))
        local_raw = generator.generate(local_prompts)
        cursor = 0
        prepared = []
        for instance, condition, agents, archive in scenarios:
            local = []
            for _agent in agents:
                answer, quote = parse_answer(local_raw[cursor])
                local.append({"answer": answer, "quote": quote, "raw": local_raw[cursor]})
                cursor += 1
            prepared.append((instance, condition, agents, archive, local))

        second_prompts = []
        second_meta = []
        coordinated = {}
        for index, (instance, condition, agents, archive, local) in enumerate(prepared):
            answers = [item["answer"] for item in local]
            coordinated[index] = {}
            for variant, flags in ablation_flags.items():
                view = (
                    with_missing_gold(instance, agents, condition)
                    if variant == "epirag_oracle"
                    else agents
                )
                state = coordinate(
                    instance.question,
                    archive,
                    view,
                    answers,
                    nli,
                    use_reliability=flags[0],
                    use_contradiction=flags[1],
                    allow_retrieval=flags[2],
                    mode=flags[3],
                )
                coordinated[index][variant] = state
                if state.get("structural"):
                    continue
                bundle_passages = []
                seen = set()
                for snippet in state["bundle"]:
                    if snippet.passage.pid in seen:
                        continue
                    seen.add(snippet.passage.pid)
                    bundle_passages.append(snippet.passage)
                if variant == "epirag_mark" and state.get("decision") == "marked":
                    second_prompts.append(
                        marked_prompt(instance.question, bundle_passages, state.get("tags") or {})
                    )
                elif variant == "epirag_mark":
                    second_prompts.append(evidence_prompt(instance.question, bundle_passages))
                elif variant == "epirag_chain":
                    second_prompts.append(bridge_entity_prompt(instance.question, bundle_passages[:4]))
                elif variant in revise_methods:
                    second_prompts.append(
                        revision_prompt(instance.question, bundle_passages[:4], answers)
                    )
                else:
                    second_prompts.append(evidence_prompt(instance.question, bundle_passages[:4]))
                second_meta.append((index, variant))

            if need_baselines:
                union = []
                seen = set()
                for group in agents:
                    for passage in group:
                        if passage.pid in seen:
                            continue
                        seen.add(passage.pid)
                        union.append(passage)
                union = order_passages(instance.question, union)[:6]
                second_prompts.append(evidence_prompt(instance.question, union))
                second_meta.append((index, "centralized"))

                shared = []
                for group in agents:
                    ranked = order_passages(instance.question, group)
                    for passage in ranked[:2]:
                        sentence = passage.text.split(". ")[0][:280]
                        shared.append(
                            type(passage)(
                                pid=passage.pid,
                                title=passage.title,
                                text=sentence,
                                gold=passage.gold,
                                year=passage.year,
                                source_id=passage.source_id,
                            )
                        )
                second_prompts.append(evidence_prompt(instance.question, shared[:6]))
                second_meta.append((index, "collab_passages"))
            else:
                union = []
                shared = []
            prepared[index] = (instance, condition, agents, archive, local, union, shared)

            if need_baselines:
                for agent, passages in enumerate(agents):
                    others = [answers[j] for j in range(3) if j != agent]
                    second_prompts.append(debate_prompt(instance.question, passages, others))
                    second_meta.append((index, f"debate_{agent}"))

        second_raw = generator.generate(second_prompts)
        grouped_raw = defaultdict(dict)
        for meta, raw in zip(second_meta, second_raw):
            grouped_raw[meta[0]][meta[1]] = raw

        chain_final = {}
        if "epirag_chain" in ablation_flags:
            third_prompts = []
            third_meta = []
            for index, (instance, _condition, agents, _archive, _local, union, _shared) in enumerate(prepared):
                bridge, _quote = parse_answer(grouped_raw[index].get("epirag_chain", ""))
                bridge = " ".join((bridge or "").split())[:80]
                pages = pool_pages(instance.question, agents, union)
                tokens = [tok for tok in bridge.split() if tok]
                if 0 < len(tokens) <= MAX_CLAIM_TOKENS and bridge.lower() not in {"unknown", "none", "n/a"}:
                    final_prompt = chain_answer_prompt(instance.question, pages, bridge)
                    decision = "chain"
                else:
                    final_prompt = evidence_prompt(instance.question, pages)
                    decision = "pool"
                state = coordinated[index]["epirag_chain"]
                bridge_pages = []
                seen_pids = set()
                for snippet in state["bundle"]:
                    if snippet.passage.pid in seen_pids:
                        continue
                    seen_pids.add(snippet.passage.pid)
                    bridge_pages.append(snippet.passage)
                bridge_text = bridge_entity_prompt(instance.question, bridge_pages[:4])
                third_prompts.append(final_prompt)
                third_meta.append(
                    (index, decision, bridge, generator.count(bridge_text) + generator.count(final_prompt))
                )
            third_raw = generator.generate(third_prompts)
            for meta, raw in zip(third_meta, third_raw):
                answer, _quote = parse_answer(raw)
                chain_final[meta[0]] = {
                    "answer": answer,
                    "decision": meta[1],
                    "bridge": meta[2],
                    "reader_tokens": meta[3],
                }

        exec_final = {}
        if "epirag_exec" in ablation_flags:
            exec_items = []
            for index, (_instance, _condition, agents, _archive, _local, _union, _shared) in enumerate(prepared):
                pages = []
                seen_pages = set()
                for group in agents:
                    for passage in group:
                        if passage.pid in seen_pages:
                            continue
                        seen_pages.add(passage.pid)
                        pages.append(passage)
                exec_items.append((_instance.question, pages))
            for index, final in enumerate(executed_answers(generator.generate, generator.count, exec_items)):
                exec_final[index] = final

        hop_final = {}
        if "epirag_hop" in ablation_flags:
            hop_items = []
            for _index, (instance, _condition, agents, _archive, _local, _union, _shared) in enumerate(prepared):
                pages = []
                seen_pages = set()
                for group in agents:
                    for passage in group:
                        if passage.pid in seen_pages:
                            continue
                        seen_pages.add(passage.pid)
                        pages.append(passage)
                hop_items.append((instance.question, pages))
            year_final = executed_answers(generator.generate, generator.count, hop_items)
            relation_final = resolve_hops(generator.generate, hop_items, generator.count)
            for index, (year_item, relation_item) in enumerate(zip(year_final, relation_final)):
                if year_item["answer"]:
                    decision = "year"
                    answer = year_item["answer"]
                elif relation_item["answer"]:
                    decision = "hop"
                    answer = relation_item["answer"]
                else:
                    decision = "witness"
                    answer = ""
                hop_final[index] = {
                    "answer": answer,
                    "decision": decision,
                    "hop_tokens": year_item["hop_tokens"] + relation_item["hop_tokens"],
                }

        derive_final = {}
        if "epirag_derive" in ablation_flags:
            derive_items = []
            for _index, (instance, _condition, agents, _archive, _local, _union, _shared) in enumerate(prepared):
                pages = []
                seen_pages = set()
                for group in agents:
                    for passage in group:
                        if passage.pid in seen_pages:
                            continue
                        seen_pages.add(passage.pid)
                        pages.append(passage)
                derive_items.append((instance.question, pages))
            for index, final in enumerate(
                resolve_derivations(generator.generate, derive_items, generator.count)
            ):
                derive_final[index] = final

        for index, (instance, condition, agents, archive, local, union, shared) in enumerate(prepared):
            answers = [item["answer"] for item in local]
            raw_map = grouped_raw[index]
            records = {}

            if need_baselines:
                top_scores = []
                for passages in agents:
                    if not passages:
                        top_scores.append(0.0)
                        continue
                    corpus = [tokenize(p.title + " " + p.text) or ["empty"] for p in passages]
                    ranker = BM25Okapi(corpus)
                    scores = ranker.get_scores(tokenize(instance.question) or ["empty"])
                    top_scores.append(float(max(scores)))
                chosen_agent = max(range(3), key=lambda i: top_scores[i])
                indep_answer = answers[chosen_agent]
                indep_evidence = [p.text for p in agents[chosen_agent]]
                records["independent"] = {
                    "answer": indep_answer,
                    "confidence": support_confidence(nli, instance.question, indep_evidence, indep_answer),
                    "messages": 0,
                    "comm_tokens": 0,
                    "retrievals": 3,
                    "triggered": False,
                }

                cent_answer, _quote = parse_answer(raw_map["centralized"])
                cent_evidence = [p.text for p in union]
                cent_text = "\n".join(p.text for p in union)
                records["centralized"] = {
                    "answer": cent_answer,
                    "confidence": support_confidence(nli, instance.question, cent_evidence, cent_answer),
                    "messages": len(union),
                    "comm_tokens": generator.count(cent_text),
                    "retrievals": 3,
                    "triggered": False,
                }

                debate_answers = []
                for agent in range(3):
                    answer, _quote = parse_answer(raw_map[f"debate_{agent}"])
                    debate_answers.append(answer)
                debate_answer = majority(debate_answers)
                voters = [i for i, ans in enumerate(debate_answers) if re_key(ans) == re_key(debate_answer)]
                debate_evidence = []
                debate_token_text = []
                for agent in voters or [0]:
                    debate_evidence.extend(p.text for p in agents[agent])
                for ans in list(answers) + debate_answers:
                    debate_token_text.append(ans)
                records["debate"] = {
                    "answer": debate_answer,
                    "confidence": support_confidence(nli, instance.question, debate_evidence, debate_answer),
                    "messages": 6,
                    "comm_tokens": generator.count("\n".join(debate_token_text)),
                    "retrievals": 3,
                    "triggered": False,
                }

                collab_answer, _quote = parse_answer(raw_map["collab_passages"])
                collab_evidence = [p.text for p in shared]
                records["collab"] = {
                    "answer": collab_answer,
                    "confidence": support_confidence(nli, instance.question, collab_evidence, collab_answer),
                    "messages": 3,
                    "comm_tokens": generator.count("\n".join(collab_evidence)),
                    "retrievals": 3,
                    "triggered": False,
                }

                vote_answer = majority(answers)
                records["vote"] = {
                    "answer": vote_answer,
                    "confidence": support_confidence(
                        nli,
                        instance.question,
                        [p.text for group in agents for p in group[:1]],
                        vote_answer,
                    ),
                    "messages": 3,
                    "comm_tokens": generator.count("\n".join(answers)),
                    "retrievals": 3,
                    "triggered": False,
                }

            for variant in ablation_flags:
                state = coordinated[index][variant]
                bundle = state["bundle"]
                structural = state.get("structural") or ""
                if variant == "epirag_exec":
                    final = exec_final[index]
                    cover_answer, _quote = parse_answer(raw_map[variant])
                    state = dict(state)
                    state["mode"] = "execute"
                    state["decision"] = final["decision"]
                    bundle_passages = []
                    seen_exec = set()
                    for snippet in bundle:
                        if snippet.passage.pid in seen_exec:
                            continue
                        seen_exec.add(snippet.passage.pid)
                        bundle_passages.append(snippet.passage)
                    state["reader_tokens"] = final["hop_tokens"] + generator.count(
                        evidence_prompt(instance.question, bundle_passages[:4])
                    )
                    if final["answer"]:
                        answer = final["answer"]
                        structural = final["answer"]
                        state["structural"] = final["answer"]
                    else:
                        answer = cover_answer
                elif variant == "epirag_hop":
                    final = hop_final[index]
                    cover_answer, _quote = parse_answer(raw_map[variant])
                    state = dict(state)
                    state["mode"] = "hop"
                    state["decision"] = final["decision"]
                    bundle_passages = []
                    seen_hop = set()
                    for snippet in bundle:
                        if snippet.passage.pid in seen_hop:
                            continue
                        seen_hop.add(snippet.passage.pid)
                        bundle_passages.append(snippet.passage)
                    state["reader_tokens"] = final["hop_tokens"] + generator.count(
                        evidence_prompt(instance.question, bundle_passages[:4])
                    )
                    if final["answer"]:
                        answer = final["answer"]
                        structural = final["answer"]
                        state["structural"] = final["answer"]
                    else:
                        answer = cover_answer
                elif variant == "epirag_derive":
                    final = derive_final[index]
                    cover_answer, _quote = parse_answer(raw_map[variant])
                    state = dict(state)
                    state["mode"] = "derive"
                    state["decision"] = final["decision"]
                    bundle_passages = []
                    seen_derive = set()
                    for snippet in bundle:
                        if snippet.passage.pid in seen_derive:
                            continue
                        seen_derive.add(snippet.passage.pid)
                        bundle_passages.append(snippet.passage)
                    state["reader_tokens"] = final["hop_tokens"] + generator.count(
                        evidence_prompt(instance.question, bundle_passages[:4])
                    )
                    if final["answer"]:
                        answer = final["answer"]
                        structural = final["answer"]
                        state["structural"] = final["answer"]
                    else:
                        answer = cover_answer
                elif structural:
                    answer = structural
                    if structural != "conflict":
                        answer = canonicalize_polar(instance.question, structural)
                elif variant == "epirag_chain":
                    final = chain_final[index]
                    answer = final["answer"]
                    state = dict(state)
                    state["decision"] = final["decision"]
                    state["hat"] = final["bridge"]
                    state["mode"] = "chain"
                    state["reader_tokens"] = final["reader_tokens"]
                else:
                    answer, _quote = parse_answer(raw_map[variant])
                passage_evidence = []
                bundle_pages = []
                seen_pids = set()
                for snippet in bundle:
                    if snippet.passage.pid in seen_pids:
                        continue
                    seen_pids.add(snippet.passage.pid)
                    passage_evidence.append(snippet.passage.text)
                    bundle_pages.append(snippet.passage)
                reader_answer = answer
                if variant in gate_methods and not structural:
                    answer = commit_answer(
                        nli,
                        instance.question,
                        passage_evidence,
                        answers,
                        reader_answer,
                    )
                elif variant in revise_methods:
                    answer = canonicalize_polar(instance.question, reader_answer)
                evidence = [snippet.text for snippet in bundle]
                weights = [
                    state["reliability"].get(snippet.passage.source_id, 1.0) for snippet in bundle
                ]
                message_text = "\n".join(
                    f"{snippet.passage.source_id}|{snippet.text}" for snippet in bundle
                )
                retrievals = 3 + (1 if state["triggered"] else 0)
                messages = 3 + (1 if state["triggered"] else 0)
                records[variant] = {
                    "answer": answer,
                    "confidence": support_confidence(
                        nli, instance.question, evidence, answer, weights
                    ),
                    "messages": messages,
                    "comm_tokens": generator.count(message_text),
                    "retrievals": retrievals,
                    "triggered": bool(state["triggered"]),
                    "hat": state["hat"],
                    "margin": state["margin"],
                    "beliefs": [round(b, 4) for b in state["beliefs"]],
                    "claims": state["claims"],
                    "coverage_gap": state.get("coverage_gap"),
                    "mode": state.get("mode", "flat"),
                    "reader_answer": reader_answer,
                    "decision": state.get("decision", "reader"),
                    "structural": structural,
                    "nli_pairs": int(state.get("nli_pairs") or 0),
                    "reader_tokens": state.get("reader_tokens")
                    if variant in {"epirag_chain", "epirag_exec", "epirag_hop", "epirag_derive"}
                    else 0
                    if structural
                    else generator.count(
                        marked_prompt(instance.question, bundle_pages, state.get("tags") or {})
                        if variant == "epirag_mark" and state.get("decision") == "marked"
                        else evidence_prompt(instance.question, bundle_pages)
                        if variant == "epirag_mark"
                        else revision_prompt(instance.question, bundle_pages, answers)
                        if variant in revise_methods
                        else evidence_prompt(instance.question, bundle_pages[:4])
                    ),
                    "local_tokens": sum(
                        generator.count(evidence_prompt(instance.question, group))
                        for group in agents
                    ),
                }

            for method, payload in records.items():
                key = (instance.dataset, instance.qid, condition, method)
                if key in done:
                    continue
                row = {
                    "dataset": instance.dataset,
                    "qid": instance.qid,
                    "qtype": instance.qtype,
                    "question": instance.question,
                    "gold": instance.answer,
                    "condition": condition,
                    "method": method,
                    "answer": payload["answer"],
                    "em": exact_match(payload["answer"], instance.answer),
                    "f1": token_f1(payload["answer"], instance.answer),
                    "confidence": payload["confidence"],
                    "messages": payload["messages"],
                    "comm_tokens": payload["comm_tokens"],
                    "retrievals": payload["retrievals"],
                    "triggered": payload["triggered"],
                    "claims": payload.get("claims", []),
                    "beliefs": payload.get("beliefs", []),
                    "hat": payload.get("hat", ""),
                    "margin": payload.get("margin", None),
                    "coverage_gap": payload.get("coverage_gap"),
                    "mode": payload.get("mode", ""),
                    "reader_answer": payload.get("reader_answer", ""),
                    "decision": payload.get("decision", ""),
                    "structural": payload.get("structural", ""),
                    "nli_pairs": payload.get("nli_pairs", None),
                    "reader_tokens": payload.get("reader_tokens", None),
                    "local_tokens": payload.get("local_tokens", None),
                    "local_answers": answers,
                }
                append_row(args.out, row)
                done.add(key)
        print(f"finished {min(start + chunk_size, len(pending))}/{len(pending)}", flush=True)

    summarize(args.out)


def summarize(path: str):
    if not os.path.exists(path):
        return
    buckets = defaultdict(lambda: defaultdict(list))
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            buckets[(row["dataset"], row["method"])][row["condition"]].append(row)
    print("\n=== summary EM ===", flush=True)
    for key in sorted(buckets):
        dataset, method = key
        parts = []
        for condition in CONDITIONS:
            rows = buckets[key][condition]
            if not rows:
                continue
            em = sum(r["em"] for r in rows) / len(rows)
            parts.append(f"{condition}={em:.3f}(n={len(rows)})")
        print(f"{dataset:10} {method:24} " + " ".join(parts), flush=True)


if __name__ == "__main__":
    run()
