"""Private views without answer leakage.

Every agent holds a small view of pages and nothing in *which* pages it holds,
or in what order, reveals the answer. Two regimes:

split       the supporting pages of the question are dealt to different
            agents (round robin from a random agent, in random order); every
            view is filled with the strongest BM25 distractors, dealt round
            robin so that views are equally hard. No single agent can
            complete a multi-hop derivation from its own view.
colocated   all supporting pages are held by one random agent; the others
            hold distractors only. Communication is not needed: this is the
            control in which a protocol must do no harm.

All randomness is seeded by the question id, the regime and a protocol seed,
so every method sees byte-identical views. Pages inside a view are shuffled:
the position of a page carries no information.

Compared with the earlier protocol (``epinet.data.views``) this removes three
leaks we measured: rewritten copies were always copies of the answer page (on
comparison questions the duplicated entity *is* the answer), the withheld
page in ``incomplete`` was always the non-answer page, and supporting pages
were always placed first in a view.
"""

from __future__ import annotations

import random

from epinet.data.examples import Example, Page
from epinet.data.views import bm25_order
from epinet.text import stable_hash

REGIMES = ("split", "colocated")


def _rng(example: Example, regime: str, seed: int) -> random.Random:
    return random.Random(int(stable_hash(f"{example.qid}|{regime}|{seed}")[:12], 16))


def build_views(example: Example, regime: str, n_agents: int = 3, k: int = 3, seed: int = 0) -> list[list[Page]]:
    rng = _rng(example, regime, seed)
    ranked = bm25_order(example.question, [p.copy() for p in example.pages])
    golds = [p for p in ranked if p.gold] or ranked[:1]
    distractors = [p for p in ranked if p.pid not in {g.pid for g in golds}]
    golds = list(golds)
    rng.shuffle(golds)
    views: list[list[Page]] = [[] for _ in range(n_agents)]
    first = rng.randrange(n_agents)
    if regime == "split":
        for i, page in enumerate(golds):
            views[(first + i) % n_agents].append(page)
    elif regime == "colocated":
        views[first].extend(golds)
    else:
        raise ValueError(regime)
    # deal distractors round robin, strongest first, to every view that still has room
    pool = iter(distractors)
    while any(len(v) < k for v in views):
        progressed = False
        for a in range((first + 1) % n_agents, (first + 1) % n_agents + n_agents):
            view = views[a % n_agents]
            if len(view) < k:
                page = next(pool, None)
                if page is None:
                    break
                view.append(page)
                progressed = True
        if not progressed:
            break
    for view in views:
        rng.shuffle(view)
    return views


def union(views: list[list[Page]], example: Example, regime: str, seed: int = 0) -> list[Page]:
    """The omniscient reference: one reader holds every page of every view."""
    pages = [p for v in views for p in v]
    _rng(example, regime + "/union", seed).shuffle(pages)
    return pages
