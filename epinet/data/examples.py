"""Unified multi-hop QA examples for HotpotQA, 2WikiMultihopQA and MuSiQue.

Every example keeps its pages as sentence lists, the supporting sentence ids,
and, where the dataset provides them, the gold intermediate answers. The
intermediate answers are only used as optional step-level supervision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from epinet.text import split_sentences


@dataclass
class Page:
    pid: str
    title: str
    sentences: list[str]
    gold: bool = False
    support: list[int] = field(default_factory=list)
    year: int = 2020
    kind: str = "orig"  # orig | corrupt | stale

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    def copy(self, **changes) -> "Page":
        return replace(self, sentences=list(self.sentences), support=list(self.support), **changes)


@dataclass
class Example:
    qid: str
    dataset: str
    question: str
    answers: list[str]
    qtype: str
    pages: list[Page]
    intermediate: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str:
        return self.answers[0]


def _hotpot_like(row: dict, dataset: str) -> Example:
    support: dict[str, list[int]] = {}
    facts = row.get("supporting_facts") or {}
    for title, sid in zip(facts.get("title", []), facts.get("sent_id", [])):
        support.setdefault(str(title), []).append(int(sid))
    pages = []
    ctx = row["context"]
    for i, (title, sents) in enumerate(zip(ctx["title"], ctx["sentences"])):
        sentences = [" ".join(str(s).split()) for s in sents if str(s).strip()]
        if not sentences:
            continue
        title = str(title)
        pages.append(
            Page(
                pid=f"p{i}",
                title=title,
                sentences=sentences,
                gold=title in support,
                support=[s for s in support.get(title, []) if s < len(sentences)],
            )
        )
    intermediate: list[str] = []
    for triple in row.get("evidences") or []:
        if len(triple) == 3:
            intermediate.extend([str(triple[0]), str(triple[2])])
    return Example(
        qid=str(row["id"]),
        dataset=dataset,
        question=" ".join(str(row["question"]).split()),
        answers=[str(row["answer"]).strip()],
        qtype=str(row.get("type", "unknown")),
        pages=pages,
        intermediate=sorted(set(intermediate)),
    )


def _musique(row: dict) -> Example:
    pages = []
    for para in row["paragraphs"]:
        sentences = split_sentences(para["paragraph_text"], min_chars=4)
        if not sentences:
            continue
        gold = bool(para.get("is_supporting"))
        pages.append(
            Page(
                pid=f"p{para['idx']}",
                title=str(para["title"]),
                sentences=sentences,
                gold=gold,
                support=list(range(len(sentences))) if gold else [],
            )
        )
    hops = [str(step.get("answer", "")) for step in row.get("question_decomposition") or []]
    n_hops = len(hops)
    return Example(
        qid=str(row["id"]),
        dataset="musique",
        question=" ".join(str(row["question"]).split()),
        answers=[str(row["answer"]).strip()] + [str(a) for a in row.get("answer_aliases") or []],
        qtype=f"{n_hops}hop",
        pages=pages,
        intermediate=sorted({h for h in hops[:-1] if h}),
    )


def load_split(dataset: str, split: str):
    from datasets import load_dataset

    if dataset == "hotpotqa":
        return load_dataset("hotpot_qa", "distractor", split=split)
    if dataset == "2wiki":
        return load_dataset("framolfese/2WikiMultihopQA", split=split)
    if dataset == "musique":
        return load_dataset("dgslibisey/MuSiQue", split=split)
    raise ValueError(dataset)


def convert(row: dict, dataset: str) -> Example:
    if dataset == "musique":
        return _musique(row)
    return _hotpot_like(row, dataset)


def sample_examples(dataset: str, split: str, n: int, seed: int, offset: int = 0) -> list[Example]:
    """Type-stratified, seeded sample. ``offset`` skips earlier blocks of the same order.

    The order is a round robin over question types after a seeded shuffle, so
    disjoint blocks (train / dev / test) are prefixes of one fixed sequence.
    """
    rows = load_split(dataset, split)
    by_type: dict[str, list[int]] = {}
    for i in range(len(rows)):
        row = rows[i]
        if dataset == "musique":
            if not row.get("answerable", True):
                continue
            key = f"{len(row.get('question_decomposition') or [])}hop"
        else:
            key = str(row.get("type", "unknown"))
        by_type.setdefault(key, []).append(i)
    rng = random.Random(seed)
    buckets = []
    for key in sorted(by_type):
        ids = list(by_type[key])
        rng.shuffle(ids)
        buckets.append(ids)
    order: list[int] = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                order.append(bucket.pop())
    chosen = order[offset : offset + n]
    return [convert(rows[i], dataset) for i in chosen]
