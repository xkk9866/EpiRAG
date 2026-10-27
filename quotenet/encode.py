"""Episodes -> tensors. One sequence per agent: the agent reads its whole view.

Layout of agent ``a``::

    [CLS] question [SEP] title_1. sentences_1 [SEP] title_2. ... [SEP]

Pages share the token budget by water filling (short pages keep all their
tokens, long pages are cut to an equal share). The omniscient reference is
the same layout with one agent that holds the union of all views.

Yes/no answers are not a separate output: two sentinel tokens ``yes`` and
``no`` follow the question in every view, so the one span pointer answers every
question and a yes/no question is a span of length one.

Labels
* ``start`` / ``end``: every occurrence of a gold answer string on any page of
  any agent, or the sentinel token for yes/no (marginal likelihood over
  occurrences);
* ``kind``: 0 span, 1 yes, 2 no (reporting only);
* ``sent`` / ``sent_label``: sentence id of each page token and a per-sentence
  supporting label (dataset supporting facts).
"""

from __future__ import annotations

import random
import re

import torch

from epinet.data.examples import Example
from epinet.text import normalize_answer
from quotenet.views import REGIMES, build_views, union

KIND = {"span": 0, "yes": 1, "no": 2}
YES_PAGE, NO_PAGE = 1000, 1001   # pseudo pages of the two sentinel tokens


def answer_kind(example: Example) -> int:
    return KIND.get(normalize_answer(example.answer), 0)


def page_text(page) -> tuple[str, list[tuple[int, int]]]:
    head = f"{page.title}. "
    spans, cursor, parts = [], len(head), [head]
    for sentence in page.sentences:
        spans.append((cursor, cursor + len(sentence)))
        parts.append(sentence + " ")
        cursor += len(sentence) + 1
    return "".join(parts), spans


def occurrences(text: str, answers: list[str]) -> list[tuple[int, int]]:
    found = []
    for ans in answers:
        ans = ans.strip()
        if not ans or normalize_answer(ans) in {"yes", "no"}:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(ans) + r"(?!\w)", text, flags=re.IGNORECASE):
            found.append((m.start(), m.end()))
    return found


def water_fill(lengths: list[int], budget: int) -> list[int]:
    share = [0] * len(lengths)
    left, todo = budget, sorted(range(len(lengths)), key=lambda i: lengths[i])
    for n, i in enumerate(todo):
        share[i] = min(lengths[i], left // (len(todo) - n))
        left -= share[i]
    return share


def encode_view(tok, question: str, pages, answers, max_len: int, max_q: int = 64):
    q_ids = tok(question, add_special_tokens=False)["input_ids"][:max_q]
    yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok("no", add_special_tokens=False)["input_ids"][0]
    gold_kind = KIND.get(normalize_answer(answers[0]), 0) if answers else 0
    encs = []
    for page in pages:
        text, spans = page_text(page)
        e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        encs.append((page, text, spans, e["input_ids"], e["offset_mapping"]))
    budget = max_len - len(q_ids) - 5 - len(pages)
    share = water_fill([len(e[3]) for e in encs], budget)
    ids = [tok.cls_token_id] + q_ids + [tok.sep_token_id, yes_id, no_id, tok.sep_token_id]
    n = len(ids)
    L = n + sum(share) + len(pages)
    tok_page = [-1] * L       # page index of every token, -1 outside pages
    tok_sent = [-1] * L       # (page, sentence) of every page token
    start, end = [False] * L, [False] * L
    chars = [None] * L        # char offsets inside the page text
    tok_page[n - 3], tok_page[n - 2] = YES_PAGE, NO_PAGE
    start[n - 3 + gold_kind - 1] = end[n - 3 + gold_kind - 1] = gold_kind > 0
    texts = []
    for p, ((page, text, spans, pids, offs), cut) in enumerate(zip(encs, share)):
        occ = occurrences(text, answers)
        for t in range(cut):
            pos = len(ids)
            ids.append(pids[t])
            c0, c1 = offs[t]
            if c1 <= c0:
                continue
            tok_page[pos] = p
            chars[pos] = (c0, c1)
            for si, (s0, s1) in enumerate(spans):
                if s0 <= c0 < s1:
                    tok_sent[pos] = (p, si)
                    break
            for o0, o1 in occ:
                if c0 <= o0 < c1:
                    start[pos] = True
                if c0 < o1 <= c1:
                    end[pos] = True
        ids.append(tok.sep_token_id)
        texts.append(text)
    return {"ids": ids, "page": tok_page, "sent": tok_sent, "start": start, "end": end,
            "chars": chars, "texts": texts, "pages": pages}


def encode_episode(tok, example: Example, regime: str, max_len: int = 512, centralize: bool = False,
                   protocol_seed: int = 0) -> dict:
    views = build_views(example, regime, seed=protocol_seed)
    if centralize:
        views = [union(views, example, regime, protocol_seed)]
        max_len = max_len * 3
    agents = [encode_view(tok, example.question, v, example.answers, max_len) for v in views]
    # sentence ids are global within the episode; labels follow the order of first appearance
    sent_index, sent_label = {}, []
    sent = []
    for a, enc in enumerate(agents):
        row = []
        for key in enc["sent"]:
            if key == -1:
                row.append(-1)
                continue
            gk = (a,) + key
            if gk not in sent_index:
                sent_index[gk] = len(sent_label)
                page = enc["pages"][key[0]]
                sent_label.append(1.0 if (page.gold and key[1] in page.support) else 0.0)
            row.append(sent_index[gk])
        sent.append(row)
    return {
        "agents": agents, "sent": sent, "sent_label": sent_label, "kind": answer_kind(example),
        "meta": {"qid": example.qid, "dataset": example.dataset, "qtype": example.qtype, "regime": regime,
                 "question": example.question, "answers": example.answers,
                 "gold_agents": [a for a, v in enumerate(views) if any(p.gold for p in v)]},
    }


def collate(items: list[dict], pad_id: int) -> dict:
    B, A = len(items), len(items[0]["agents"])
    L = max(len(ag["ids"]) for it in items for ag in it["agents"])
    ids = torch.full((B, A, L), pad_id, dtype=torch.long)
    mask = torch.zeros((B, A, L), dtype=torch.bool)
    page = torch.full((B, A, L), -1, dtype=torch.long)
    start = torch.zeros((B, A, L), dtype=torch.bool)
    end = torch.zeros((B, A, L), dtype=torch.bool)
    sent = torch.full((B, A, L), -1, dtype=torch.long)
    labels, offset = [], 0
    for b, it in enumerate(items):
        for a, ag in enumerate(it["agents"]):
            n = len(ag["ids"])
            ids[b, a, :n] = torch.tensor(ag["ids"])
            mask[b, a, :n] = True
            page[b, a, :n] = torch.tensor(ag["page"])
            start[b, a, :n] = torch.tensor(ag["start"])
            end[b, a, :n] = torch.tensor(ag["end"])
            s = torch.tensor(it["sent"][a])
            s[s >= 0] += offset
            sent[b, a, :n] = s
        labels.append(torch.tensor(it["sent_label"]))
        offset += len(it["sent_label"])
    return {
        "ids": ids, "mask": mask, "page": page, "ctx": page >= 0, "qok": (page >= 0) & (page < 1000), "start": start, "end": end,
        "sent": sent, "sent_label": torch.cat(labels) if labels else torch.zeros(0),
        "kind": torch.tensor([it["kind"] for it in items]),
        "meta": [it["meta"] for it in items],
        "views": [[{"chars": ag["chars"], "texts": ag["texts"], "ids": ag["ids"]} for ag in it["agents"]] for it in items],
    }


class Episodes(torch.utils.data.Dataset):
    """Training draws a fresh regime each time an example is visited."""

    def __init__(self, tok, examples: list[Example], max_len: int = 512, centralize: bool = False,
                 regimes=REGIMES, fixed: list[str] | None = None, seed: int = 0, protocol_seed: int = 0):
        self.tok, self.examples, self.max_len, self.centralize = tok, examples, max_len, centralize
        self.regimes, self.fixed = list(regimes), fixed
        self.rng = random.Random(seed)
        self.protocol_seed = protocol_seed

    def __len__(self):
        return len(self.examples) * (len(self.fixed) if self.fixed else 1)

    def __getitem__(self, i):
        if self.fixed:
            ex, regime = self.examples[i // len(self.fixed)], self.fixed[i % len(self.fixed)]
            ps = self.protocol_seed
        else:  # training: a fresh deal of the same question at every visit
            ex, regime = self.examples[i], self.rng.choice(self.regimes)
            ps = self.rng.randrange(1 << 30)
        return encode_episode(self.tok, ex, regime, self.max_len, self.centralize, ps)
