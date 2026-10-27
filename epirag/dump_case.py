from datasets import load_dataset

from epirag.data import build_instance, select_examples

wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
_, eval_ids = select_examples(wiki, "2wiki", 1, 12, 7)
inst = build_instance(wiki[eval_ids[0]], "2wiki")
print("Q:", inst.question)
print("A:", inst.answer)
print("--- CORRUPTION VIEWS ---")
mapping = inst.passage_map()
for agent, pids in enumerate(inst.views["corruption"]):
    print(f"\nAGENT {agent}")
    for pid in pids:
        p = mapping[pid]
        print(f"  [{p.pid}] {p.title} | year={p.year} | gold={p.gold}")
        print("   ", p.text[:280].replace("\n", " "))
