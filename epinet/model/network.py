"""EpiNet: dependence-aware neural belief propagation over private views.

Every agent is the same reader applied to its own pages. Its belief is a
distribution over answer tokens in its view. Agents then run ``T`` rounds of
one shared message-passing step:

    opinion   o_a = sum_t p_a(t) h_t              (expected answer embedding)
    message   z_a ~ N(mu(o_a, e_a, c_a), sigma)   (budgeted Gaussian message)
    weigh     w_ab = trust(z_a, z_b) * indep(e_a, e_b)
    corrob.   l_a(t) <- l_a(t) + beta * sum_b w_ab <U h_t, V z_b>
    bridge    h_t    <- h_t + gamma * CrossAttn(h_t, {w_ab z_b})

``e_a`` is a fingerprint of the evidence agent ``a`` read. A product of
experts double counts agents that read the same evidence; ``indep`` is a
learned kernel on evidence fingerprints that removes that double counting, so
a correlated majority (two agents holding one rewritten copy) carries the
weight of one source, not two. ``beta`` and ``gamma`` start at zero: at
initialisation the network is exactly a set of independent readers, and the
channel is used only to the extent answer supervision rewards it.

The answer is the argmax over every token of every view of the final logits.
"""

from __future__ import annotations

from dataclasses import dataclass

from epinet.model.quant import quantize_module

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetConfig:
    encoder: str = "microsoft/deberta-v3-base"
    slots: int = 4            # kept for checkpoint compatibility; unused
    rounds: int = 2
    heads: int = 8
    dropout: float = 0.1
    channel: bool = True      # ablation: agents never exchange messages
    write_back: bool = True   # ablation: no bridge (messages do not re-encode evidence)
    corroborate: bool = True  # ablation: no corroboration term on answer logits
    independence: bool = True # ablation: indep(e_a, e_b) fixed to 1
    max_span: int = 30
    grad_ckpt: bool = True
    rate_weight: float = 1e-3


class Block(nn.Module):
    """Pre-norm cross-attention + feed-forward."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.nq = nn.LayerNorm(d)
        self.nk = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.nf = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, q, kv, key_pad=None, attn_bias=None):
        k = self.nk(kv)
        out, _ = self.attn(self.nq(q), k, k, key_padding_mask=key_pad, attn_mask=attn_bias, need_weights=False)
        q = q + out
        return q + self.ff(self.nf(q))


def _masked_mean(x, m):
    m = m.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class EpiNet(nn.Module):
    def __init__(self, cfg: NetConfig, encoder: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(cfg.encoder)
        self.encoder = encoder
        if cfg.grad_ckpt and hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        d = encoder.config.hidden_size
        self.d = d
        self.span = nn.Linear(d, 2)                     # local reader: start/end
        self.msg = nn.Sequential(nn.Linear(3 * d + 1, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.trust_q = nn.Linear(d, d)
        self.trust_k = nn.Linear(d, d)
        self.indep = nn.Sequential(nn.Linear(2 * d, d // 4), nn.GELU(), nn.Linear(d // 4, 1))
        self.cor_tok = nn.Linear(d, 2 * d)              # U for start and end
        self.cor_msg = nn.Linear(d, 2 * d)              # V for start and end
        self.bridge = Block(d, cfg.heads, cfg.dropout)
        self.beta = nn.Parameter(torch.zeros(2))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.kind = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))
        self.support = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        # INT8 quantization-aware training: every linear map and attention product
        # runs on INT8 codes (see quant.py).
        quantize_module(self)

    def encode(self, ids, mask):
        B, A, P, L = ids.shape
        flat = self.encoder(input_ids=ids.view(-1, L), attention_mask=mask.view(-1, L).long()).last_hidden_state
        return flat.view(B, A, P, L, self.d)

    def forward(self, ids, mask, ctx, sent=None):
        B, A, P, L = ids.shape
        d, N = self.d, P * L
        H = self.encode(ids, mask)
        X = H.reshape(B * A, N, d)
        valid = ctx.reshape(B * A, N)
        neg = -1e4
        cls = H[:, :, :, 0].mean(2).reshape(B * A, d)
        e = F.normalize(_masked_mean(X, valid).float(), dim=-1).to(X.dtype)  # evidence fingerprint

        logits = self.span(X).float()                    # [BA, N, 2]
        rate = X.new_zeros((), dtype=torch.float)
        weights = None
        use = self.cfg.channel and A > 1
        if use:
            off = ~torch.eye(A, dtype=torch.bool, device=ids.device)
            if self.cfg.independence:
                ea = e.view(B, A, 1, d).expand(B, A, A, d)
                eb = e.view(B, 1, A, d).expand(B, A, A, d)
                dep_in = torch.cat([ea * eb, (ea - eb).abs()], -1)
                indep = torch.sigmoid(self.indep(dep_in).squeeze(-1).float())  # [B, A, A]
            else:
                indep = torch.ones(B, A, A, device=ids.device)
            for _ in range(self.cfg.rounds):
                ls = logits[..., 0].masked_fill(~valid, neg)
                p = torch.softmax(ls, -1).to(X.dtype)
                o = torch.einsum("bn,bnd->bd", p, X)                     # opinion
                conf = torch.logsumexp(ls, -1, keepdim=True).to(X.dtype)  # evidential mass
                mu, logvar = self.msg(torch.cat([o, e, cls, conf / 10.0], -1)).chunk(2, -1)
                logvar = logvar.float().clamp(-8.0, 8.0)
                mu_f = mu.float()
                z = (mu_f + torch.randn_like(mu_f) * (0.5 * logvar).exp()) if self.training else mu_f
                rate = rate + 0.5 * (mu_f.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1).mean()
                z = z.to(X.dtype).view(B, A, d)
                q = self.trust_q(z).float()
                k = self.trust_k(z).float()
                trust = torch.sigmoid(torch.einsum("bad,bcd->bac", q, k) * d ** -0.5)
                w = trust * indep * off                                  # [B, A, A]
                weights = w
                # corroboration: tokens resembling what trusted, independent peers believe
                inbox = torch.einsum("bac,bcd->bad", w.to(X.dtype), z)   # [B, A, d]
                V = self.cor_msg(inbox).view(B * A, 1, 2, d)
                U = self.cor_tok(X).view(B * A, N, 2, d)
                cor = (U * V).sum(-1).float() * d ** -0.5                # [BA, N, 2]
                # bridge: peers' messages re-encode own evidence (e.g. a bridge entity)
                if self.cfg.write_back:
                    msgs = (w.unsqueeze(-1).to(X.dtype) * z.unsqueeze(1)).reshape(B * A, A, d)
                    X = X + torch.tanh(self.gamma).to(X.dtype) * (self.bridge(X, msgs) - X)
                logits = self.span(X).float()
                if self.cfg.corroborate:
                    logits = logits + self.beta.float() * cor

        start = logits[..., 0].masked_fill(~valid, neg).view(B, -1)
        end = logits[..., 1].masked_fill(~valid, neg).view(B, -1)
        ps = torch.softmax(start, -1).view(B * A, N).to(X.dtype)
        o = torch.einsum("bn,bnd->bd", ps, X).view(B, A, d)
        kind_in = torch.cat([o.sum(1), cls.view(B, A, d).mean(1)], -1)
        out = {"start": start, "end": end, "kind": self.kind(kind_in),
               "rate": rate / max(self.cfg.rounds, 1)}
        if weights is not None:
            out["weights"] = weights.detach()
        if sent is not None:
            flat_sent = sent.view(-1)
            keep = flat_sent >= 0
            n_sent = int(flat_sent.max().item()) + 1 if keep.any() else 0
            tok_logit = self.support(X.reshape(-1, d)).squeeze(-1)
            summed = torch.zeros(n_sent, device=ids.device, dtype=tok_logit.dtype)
            count = torch.zeros(n_sent, device=ids.device, dtype=tok_logit.dtype)
            summed.index_add_(0, flat_sent[keep], tok_logit[keep])
            count.index_add_(0, flat_sent[keep], torch.ones_like(tok_logit[keep]))
            out["support"] = summed / count.clamp(min=1)
        return out


def loss_fn(out: dict, batch: dict, support_weight: float = 0.5, rate_weight: float = 1e-3) -> tuple[torch.Tensor, dict]:
    B = batch["ids"].shape[0]
    kind_loss = F.cross_entropy(out["kind"].float(), batch["kind"])
    gold_s = batch["start"].view(B, -1)
    gold_e = batch["end"].view(B, -1)
    has = (batch["kind"] == 0) & gold_s.any(1) & gold_e.any(1)
    span_loss = out["start"].new_zeros(())
    if has.any():
        ls = torch.log_softmax(out["start"][has].float(), -1)
        le = torch.log_softmax(out["end"][has].float(), -1)
        span_loss = -(torch.logsumexp(ls.masked_fill(~gold_s[has], -1e4), -1)
                      + torch.logsumexp(le.masked_fill(~gold_e[has], -1e4), -1)).mean()
    sup_loss = out["start"].new_zeros(())
    if "support" in out and out["support"].numel():
        sup_loss = F.binary_cross_entropy_with_logits(out["support"].float(), batch["sent_label"].to(out["support"].device))
    rate = out.get("rate", out["start"].new_zeros(()))
    total = span_loss + kind_loss + support_weight * sup_loss + rate_weight * rate
    return total, {"span": float(span_loss), "kind": float(kind_loss), "support": float(sup_loss), "rate": float(rate)}


@torch.no_grad()
def decode(out: dict, batch: dict, max_span: int = 30) -> list[dict]:
    """Best span inside one page, or yes/no, per episode."""
    B, A, P, L = batch["ids"].shape
    results = []
    kinds = out["kind"].argmax(-1).tolist()
    tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=out["start"].device)) & ~torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=out["start"].device), diagonal=max_span
    )
    for b in range(B):
        st = out["start"][b].float().view(A * P, L)
        en = out["end"][b].float().view(A * P, L)
        scores = (st.unsqueeze(2) + en.unsqueeze(1)).masked_fill(~tri, float("-inf"))
        flat = scores.view(-1).argmax().item()
        row, rem = divmod(flat, L * L)
        i, j = divmod(rem, L)
        a, p = divmod(row, P)
        meta = batch["meta"][b]
        text, offsets, _seg = meta["texts"].get((a, p), ("", [], []))
        span = ""
        if offsets and i < len(offsets) and j < len(offsets):
            span = text[offsets[i][0]: offsets[j][1]].strip()
        answer = {1: "yes", 2: "no"}.get(kinds[b], span)
        results.append({"answer": answer, "agent": a, "page": p, "span": span})
    return results
