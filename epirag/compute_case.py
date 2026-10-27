"""Belief state for the overview worked example. Uses GPU 0 so the eval can keep GPU 1."""

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.engine import NLI, coordinate
from epirag.run_eval import materialize

wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
_, eval_ids = select_examples(wiki, "2wiki", 1, 12, 7)
inst = build_instance(wiki[eval_ids[0]], "2wiki")
agents, archive = materialize(inst, "corruption")
claims = ["Scotland", "British", "Fiona Bruce"]
nli = NLI("cross-encoder/nli-deberta-v3-base", "cuda:0")
for mode in ("cover", "flat", "drop"):
    state = coordinate(inst.question, archive, agents, claims, nli, mode=mode)
    print("MODE", mode)
    print("Q", inst.question)
    print("hat", state["hat"], "margin", round(state["margin"], 3), "trig", state["triggered"])
    for claim, belief in zip(state["claims"], state["beliefs"]):
        print(f"  {belief:.3f}  {claim}")
    print("bundle:")
    for snippet in state["bundle"]:
        print(
            f"  r={state['reliability'].get(snippet.passage.source_id, 0):.2f} "
            f"{snippet.passage.title[:28]} | {snippet.text[:140]}"
        )
    print("reliability", {k: round(v, 3) for k, v in state["reliability"].items()})
