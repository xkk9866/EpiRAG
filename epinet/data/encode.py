"""Turn (example, condition) episodes into padded tensors for the network.

Layout: ``[agents, pages, tokens]``. Each page is encoded as
``question [SEP] title (date). sentence sentence ...`` so that the date, the
only recency signal, is ordinary text the network reads; nothing downstream
treats it specially.

Labels:
* ``start`` / ``end``: every occurrence of a gold answer string in any page of
  any view (marginal likelihood over occurrences; a rewritten copy whose
  answer was replaced simply has no occurrence);
* ``kind``: 0 span, 1 yes, 2 no;
* ``support``: per-token sentence id and a per-sentence supporting label.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

from epinet.data.examples import Example
from epinet.data.views import CONDITIONS, build_views
from epinet.text import normalize_answer

KIND = {"span": 0, "yes": 1, "no": 2}


@dataclass
class EncodeConfig:
    max_len: int = 256
    n_agents: int = 3
    pages_per_agent: int = 3
    centralize: bool = False  # reference: one reader holds the union of all views


def answer_kind(example: Example) -> int:
    key = normalize_answer(example.answer)
    return KIND.get(key, 0)


def _page_text(page) -> tuple[str, list[tuple[int, int]]]:
    head = f"{page.title} ({page.year}). "
    spans, cursor, parts = [], len(head), [head]
    for sentence in page.sentences:
        spans.append((cursor, cursor + len(sentence)))
        parts.append(sentence + " ")
        cursor += len(sentence) + 1
    return "".join(parts), spans


def _occurrences(text: str, answers: list[str]) -> list[tuple[int, int]]:
    found = []
    for ans in answers:
        ans = ans.strip()
        if not ans or normalize_answer(ans) in {"yes", "no"}:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(ans) + r"(?!\w)", text, flags=re.IGNORECASE):
            found.append((m.start(), m.end()))
    return found


def encode_episode(tokenizer, example: Example, condition: str, cfg: EncodeConfig) -> dict:
    views = build_views(example, condition, 3, 3)
    if cfg.centralize:
        union, seen = [], set()
        for view in views:
            for page in view:
                if (page.pid, page.text) not in seen:
                    seen.add((page.pid, page.text))
                    union.append(page)
        views = [union]
    A, P, L = len(views), cfg.pages_per_agent, cfg.max_len
    ids = torch.full((A, P, L), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((A, P, L), dtype=torch.bool)
    ctx = torch.zeros((A, P, L), dtype=torch.bool)  # tokens that may be part of an answer
    start = torch.zeros((A, P, L), dtype=torch.bool)
    end = torch.zeros((A, P, L), dtype=torch.bool)
    sent = torch.full((A, P, L), -1, dtype=torch.long)  # sentence slot per token
    sent_label = []
    texts = {}
    slot = 0
    for a, view in enumerate(views):
        for p, page in enumerate(view[:P]):
            text, spans = _page_text(page)
            enc = tokenizer(
                example.question, text, truncation="only_second", max_length=L,
                return_offsets_mapping=True,
            )
            seq = torch.tensor(enc["input_ids"])
            n = len(seq)
            ids[a, p, :n] = seq
            mask[a, p, :n] = True
            seg = enc.sequence_ids()
            offsets = enc["offset_mapping"]
            occ = _occurrences(text, example.answers)
            # a slot exists only for a sentence that keeps at least one token
            # after truncation, so labels and model outputs stay aligned
            page_slots = [None] * len(spans)
            for t in range(n):
                if seg[t] != 1:
                    continue
                c0, c1 = offsets[t]
                if c1 <= c0:
                    continue
                ctx[a, p, t] = True
                for si, (s0, s1) in enumerate(spans):
                    if s0 <= c0 < s1:
                        if page_slots[si] is None:
                            page_slots[si] = slot
                            sent_label.append(1.0 if (page.gold and si in page.support) else 0.0)
                            slot += 1
                        sent[a, p, t] = page_slots[si]
                        break
                for o0, o1 in occ:
                    if c0 <= o0 < c1:
                        start[a, p, t] = True
                    if c0 < o1 <= c1:
                        end[a, p, t] = True
            texts[(a, p)] = (text, offsets, seg)
    return {
        "ids": ids, "mask": mask, "ctx": ctx, "start": start, "end": end,
        "sent": sent, "sent_label": torch.tensor(sent_label, dtype=torch.float),
        "kind": answer_kind(example),
        "meta": {
            "qid": example.qid, "dataset": example.dataset, "qtype": example.qtype,
            "condition": condition, "question": example.question, "answers": example.answers,
            "texts": texts,
        },
    }


def collate(items: list[dict]) -> dict:
    out = {k: torch.stack([it[k] for it in items]) for k in ("ids", "mask", "ctx", "start", "end")}
    # Sentence slots are local to each episode; offset them into one index space.
    sent = []
    labels = []
    offset = 0
    for it in items:
        s = it["sent"].clone()
        s[s >= 0] += offset
        sent.append(s)
        labels.append(it["sent_label"])
        offset += len(it["sent_label"])
    out["sent"] = torch.stack(sent)
    out["sent_label"] = torch.cat(labels) if labels else torch.zeros(0)
    out["kind"] = torch.tensor([it["kind"] for it in items])
    out["meta"] = [it["meta"] for it in items]
    return out


class EpisodeDataset(torch.utils.data.Dataset):
    """Training draws a fresh condition each time an example is visited."""

    def __init__(self, tokenizer, examples: list[Example], cfg: EncodeConfig,
                 conditions=CONDITIONS, fixed: list[str] | None = None, seed: int = 0):
        self.tok = tokenizer
        self.examples = examples
        self.cfg = cfg
        self.conditions = list(conditions)
        self.fixed = fixed
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.examples) * (len(self.fixed) if self.fixed else 1)

    def __getitem__(self, i):
        if self.fixed:
            ex = self.examples[i // len(self.fixed)]
            cond = self.fixed[i % len(self.fixed)]
        else:
            ex = self.examples[i]
            cond = self.rng.choice(self.conditions)
        return encode_episode(self.tok, ex, cond, self.cfg)
