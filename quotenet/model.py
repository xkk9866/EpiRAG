"""QuoteNet: agents communicate by quoting their private evidence.

Every agent reads its whole private view with one shared encoder. It then
discloses a *quote*: the ``k`` tokens of its view that its pointer ranks
highest, copied verbatim and in their original order. Every agent re-reads
its own view with the peers' quotes placed in its input, so the quote enters
the receiver in the encoder's native input space (word embeddings) and every
layer can bind a peer's words to the receiver's own tokens.

    read     H_a = Enc(q, V_a)
    point    u_a(t) = w^T LN(h_t)                 (one linear pointer, tied to the
                                                   supporting-fact head)
    quote    S_a = top-k_t u_a(t) (+ Gumbel noise while training)
             e_t = n + s_t (E[x_t] - n),  s_t = 1 + p_t - stopgrad(p_t)
    re-read  H'_a = Enc(q, [e_t]_{t in S_b, b != a}, V_a)

``E`` is the encoder's own word-embedding table and ``n`` a learned null
embedding. In the forward pass ``s_t = 1``, so the receiver reads exactly the
quoted words. In the backward pass ``p_t = sigmoid(u~_t - tau)`` (``tau`` the
top-k threshold of the perturbed scores) carries the receivers' answer loss
into the pointer: what an agent discloses is learned from what its peers
needed, which is the only signal that identifies the pointer on views that
hold no answer (see the paper, Section 3).

The team answer is the best span over every agent's own view, or yes/no from
the log-linear pool of the agents' own type beliefs; every variant shares this
decision, and the only thing that crosses a view boundary is the quote.

Controlled variants change exactly one factor:
  indep    k = 0, no re-reading
  quote    QuoteNet
  belief   the pointer is the agent's own answer belief (start + end logits)
  slots    k latent vectors (attention pooling under the same pointer) instead
           of k quoted words: the message leaves the encoder's input space
  sg       no receiver gradient into the pointer (s_t = 1): the pointer is
           trained by supporting-fact labels only
  free     an untied pointer with no supporting-fact supervision: the pointer
           is trained by the receivers only
  reread   the same two passes and losses with every quote masked out: extra
           depth without communication
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = ("indep", "quote", "belief", "slots", "sg", "free", "reread")
NEG = -1e4


@dataclass
class QuoteConfig:
    encoder: str = "microsoft/deberta-v3-base"
    mode: str = "quote"
    k: int = 32
    rounds: int = 1
    max_span: int = 30
    aux_weight: float = 0.5


def gumbel_like(x):
    u = torch.rand_like(x).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(u))


class QuoteNet(nn.Module):
    def __init__(self, cfg: QuoteConfig, encoder: nn.Module | None = None, sep_id: int = 2, null_id: int | None = None):
        super().__init__()
        assert cfg.mode in MODES, cfg.mode
        self.cfg = cfg
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(cfg.encoder)
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.d = d
        self.span = nn.Linear(d, 2)
        self.support = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.sep_id = sep_id
        if cfg.mode != "indep":
            emb = encoder.get_input_embeddings().weight
            init = emb[null_id].detach().clone() if null_id is not None else emb.detach().mean(0)
            self.null = nn.Parameter(init)
        if cfg.mode == "free":
            self.pointer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        if cfg.mode == "slots":
            e = emb.shape[1]
            self.queries = nn.Parameter(torch.randn(cfg.k, d) * d ** -0.5)
            self.key = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
            self.write = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, e))

    # ------------------------------------------------------------------ reading
    def read(self, ids, mask, msgs=None, msg_mask=None):
        if msgs is None:
            return self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        emb = self.encoder.get_input_embeddings()(ids)
        S = msgs.shape[1]
        x = torch.cat([emb[:, :1], msgs.to(emb.dtype), emb[:, 1:]], 1)
        m = torch.cat([mask[:, :1], msg_mask, mask[:, 1:]], 1)
        h = self.encoder(inputs_embeds=x, attention_mask=m.long()).last_hidden_state
        return torch.cat([h[:, :1], h[:, 1 + S:]], 1)

    def heads(self, X, ctx, B, A, sent):
        logits = self.span(X).float()
        sup = self.support(X).squeeze(-1).float()
        out = {"start": logits[..., 0].masked_fill(~ctx, NEG).reshape(B, -1),
               "end": logits[..., 1].masked_fill(~ctx, NEG).reshape(B, -1),
               "_logits": logits, "_sup": sup, "_X": X}
        if sent is not None:
            flat = sent.reshape(-1)
            keep = flat >= 0
            n = int(flat.max().item()) + 1 if keep.any() else 0
            tok = sup.reshape(-1)
            summed = torch.zeros(n, device=X.device).index_add_(0, flat[keep], tok[keep])
            count = torch.zeros(n, device=X.device).index_add_(0, flat[keep], torch.ones_like(tok[keep]))
            out["support"] = summed / count.clamp(min=1)
        return out

    # ------------------------------------------------------------------ quoting
    def pointer_scores(self, cur):
        mode = self.cfg.mode
        if mode == "belief":
            return cur["_logits"][..., 0] + cur["_logits"][..., 1]
        if mode == "free":
            return self.pointer(cur["_X"]).squeeze(-1).float()
        return cur["_sup"]

    def quote(self, ids, ctx, score, probe=None):
        """Top-k tokens of each view in original order, with a straight-through gate."""
        k = self.cfg.k
        g = score.float()
        if probe == "random":
            g = torch.rand_like(g)
        elif self.training:
            g = g + gumbel_like(g)
        g = g.masked_fill(~ctx, NEG)
        top = g.topk(min(k + 1, g.shape[1]), dim=-1)
        kk = min(k, top.values.shape[1] - 1)
        thr = 0.5 * (top.values[:, kk - 1] + top.values[:, kk])
        idx, order = top.indices[:, :kk].sort(-1)
        valid = (top.values[:, :kk] > NEG / 2).gather(-1, order)
        word = self.encoder.get_input_embeddings()(ids.gather(1, idx))
        if self.cfg.mode == "sg" or not self.training:
            s = torch.ones_like(idx, dtype=word.dtype)
        else:
            p = torch.sigmoid(g.gather(1, idx) - thr.unsqueeze(1))
            s = (1.0 + p - p.detach()).to(word.dtype)
        null = self.null.to(word.dtype)
        e = null + s.unsqueeze(-1) * (word - null)
        return e, valid, idx

    def slots(self, X, ctx, score):
        k = self.key(X)
        att = torch.einsum("md,nld->nml", self.queries.to(k.dtype), k).float() * self.d ** -0.5
        att = (att + F.logsigmoid(score).unsqueeze(1)).masked_fill(~ctx.unsqueeze(1), NEG)
        m = torch.einsum("nml,nld->nmd", torch.softmax(att, -1).to(X.dtype), X)
        valid = ctx.any(1, keepdim=True).expand(-1, self.cfg.k)
        return self.write(m), valid, None

    def route(self, e, valid, B, A, probe=None):
        """Receiver ``a`` reads the quotes of every other agent, each followed by [SEP]."""
        k, de = e.shape[1], e.shape[2]
        sep = self.encoder.get_input_embeddings().weight[self.sep_id].to(e.dtype)
        e = torch.cat([e, sep.expand(e.shape[0], 1, de)], 1).view(B, A, k + 1, de)
        v = torch.cat([valid, valid.any(1, keepdim=True)], 1).view(B, A, k + 1)
        if probe == "swap":                      # counterfactual: quotes from another episode
            e, v = e.roll(1, dims=0), v.roll(1, dims=0)
        others = torch.tensor([[b for b in range(A) if b != a] for a in range(A)], device=e.device)
        msgs = e[:, others].reshape(B * A, (A - 1) * (k + 1), de)
        mmask = v[:, others].reshape(B * A, (A - 1) * (k + 1))
        if probe == "drop" or self.cfg.mode == "reread":
            mmask = torch.zeros_like(mmask)
        return msgs, mmask

    def forward(self, ids, mask, ctx, sent=None, probe=None, qok=None):
        B, A, L = ids.shape
        qok = ctx if qok is None else qok
        ids_f, mask_f, ctx_f = ids.view(B * A, L), mask.view(B * A, L), ctx.view(B * A, L)
        qok_f = qok.view(B * A, L)
        X = self.read(ids_f, mask_f)
        cur = self.heads(X, ctx_f, B, A, sent)
        rounds, quotes = [cur], []
        if self.cfg.mode == "indep" or A == 1:
            return {"final": cur, "rounds": rounds, "quotes": quotes}
        store_m, store_v = [], []
        for _ in range(self.cfg.rounds):
            score = self.pointer_scores(cur)
            if self.cfg.mode == "slots":
                e, valid, idx = self.slots(cur["_X"], qok_f, score)
            else:
                e, valid, idx = self.quote(ids_f, qok_f, score, probe)
            msgs, mmask = self.route(e, valid, B, A, probe)
            store_m.append(msgs); store_v.append(mmask)      # quotes accumulate over rounds
            X = self.read(ids_f, mask_f, torch.cat(store_m, 1), torch.cat(store_v, 1))
            cur = self.heads(X, ctx_f, B, A, sent)
            rounds.append(cur)
            quotes.append(None if idx is None else (idx.view(B, A, -1), valid.view(B, A, -1)))
        return {"final": cur, "rounds": rounds, "quotes": quotes}


def answer_loss(out: dict, batch: dict, support_weight: float = 0.5):
    """Marginal likelihood of all gold occurrences under one softmax over every view."""
    B = batch["ids"].shape[0]
    gs, ge = batch["start"].view(B, -1), batch["end"].view(B, -1)
    has = gs.any(1) & ge.any(1)
    span = out["start"].new_zeros(())
    if has.any():
        ls = torch.log_softmax(out["start"][has].float(), -1)
        le = torch.log_softmax(out["end"][has].float(), -1)
        span = -(torch.logsumexp(ls.masked_fill(~gs[has], -1e4), -1) + torch.logsumexp(le.masked_fill(~ge[has], -1e4), -1)).mean()
    sup = out["start"].new_zeros(())
    if "support" in out and out["support"].numel():
        sup = F.binary_cross_entropy_with_logits(out["support"].float(), batch["sent_label"].to(out["support"].device))
    return span + support_weight * sup, {"span": float(span), "support": float(sup)}


def quote_loss(out: dict, batch: dict, aux_weight: float = 0.5):
    total, parts = answer_loss(out["final"], batch)
    if len(out["rounds"]) > 1:
        aux = sum(answer_loss(r, batch)[0] for r in out["rounds"][:-1]) / (len(out["rounds"]) - 1)
        total = total + aux_weight * aux
        parts["aux"] = float(aux)
    return total, parts


@torch.no_grad()
def decode(out: dict, batch: dict, max_span: int = 30) -> list[dict]:
    """Best span inside one page of one agent, or yes/no, per episode."""
    B, A, L = batch["ids"].shape
    dev = out["start"].device
    band = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev)) & ~torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=dev), diagonal=max_span)
    page = batch["page"].to(dev)
    res = []
    for b in range(B):
        st = out["start"][b].float().view(A, L)
        en = out["end"][b].float().view(A, L)
        ok = band & (page[b].unsqueeze(2) == page[b].unsqueeze(1)) & (page[b] >= 0).unsqueeze(2)
        scores = (st.unsqueeze(2) + en.unsqueeze(1)).masked_fill(~ok, float("-inf"))
        flat = scores.view(-1).argmax().item()
        a, rem = divmod(flat, L * L)
        i, j = divmod(rem, L)
        view = batch["views"][b][a]
        p = int(page[b, a, i])
        span = ""
        if p >= 1000:
            span = "yes" if p == 1000 else "no"
        elif 0 <= p < len(view["texts"]) and view["chars"][i] and view["chars"][j]:
            span = view["texts"][p][view["chars"][i][0]: view["chars"][j][1]].strip()
        res.append({"answer": span, "agent": a, "span": span})
    return res
