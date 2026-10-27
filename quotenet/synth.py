"""Chain worlds: a controlled test of what must cross a view boundary.

A question asks for the end of an H-hop chain of relations that starts at a
named entity, e.g. "Who is the father of the director of Narvik?". Every fact
is its own page ("The director of Narvik is Li Yang."), and each page of the
chain lies in a different view. The other pages are decoys that use the same
relations and the same name pool, so a view holding a later link has many
equally plausible subjects to attach it to. Names are random syllable strings.
"""
from __future__ import annotations

import random

from epinet.data.examples import Example, Page

RELS = ["director", "father", "mother", "founder", "successor", "mentor", "publisher", "owner", "patron", "rival"]
ONS = ["b", "d", "k", "l", "m", "n", "r", "s", "t", "v", "z", "br", "kr", "tr", "sh", "ch"]
VOW = ["a", "e", "i", "o", "u", "an", "or", "el", "in"]


def name(rng):
    return "".join(rng.choice(ONS) + rng.choice(VOW) for _ in range(rng.choice((2, 3)))).capitalize()


def make(hops: int, i: int, seed: int, n_decoy_pages: int = 6) -> Example:
    rng = random.Random(hash((hops, i, seed)) & 0xFFFFFFFF)
    names = set()
    def fresh():
        while True:
            n = name(rng)
            if n not in names:
                names.add(n)
                return n
    rels = rng.sample(RELS, hops)
    ents = [fresh() for _ in range(hops + 1)]
    pages, k = [], 0
    def page(subj, rel, obj, gold):
        nonlocal k
        k += 1
        return Page(pid=f"s{hops}_{i}_{k}", title=subj, sentences=[f"The {rel} of {subj} is {obj}."], gold=gold, support=[0] if gold else [])
    for h in range(hops):
        pages.append(page(ents[h], rels[h], ents[h + 1], True))
    for _ in range(3 * 3 - hops):       # decoys use the chain's own relations: a relation never identifies a link
        h = rng.randrange(hops)
        subj = fresh() if rng.random() < 0.4 else rng.choice([e for j, e in enumerate(ents[:-1]) if j != h])
        pages.append(page(subj, rels[h], fresh(), False))
    q = f"Who is the {rels[-1]} of " + "".join(f"the {r} of " for r in rels[-2::-1]) + f"{ents[0]}?"
    return Example(qid=f"synth{hops}_{i}_{seed}", dataset=f"synth{hops}", question=q, answers=[ents[-1]], qtype=f"{hops}hop", pages=pages)


def sample(dataset: str, split: str, n: int, seed: int, offset: int = 0):
    hops = int(dataset[5:])
    base = {"train": 0, "validation": 10**6}[split]
    return [make(hops, base + offset + j, seed if split == "train" else 7) for j in range(n)]
