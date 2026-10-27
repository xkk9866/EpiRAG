"""Evidence Relay: agents communicate by having their evidence re-read by peers.

Belief-space protocols (vote, debate, the EpiNet belief channel) let agent ``a``
tell agent ``b`` *what it thinks the answer is*. For compositional and
comparison questions whose facts are split across private views, that
message carries no usable information: the answer is a function of the
conjunction of facts, not of any single agent's belief over answers.

Evidence Relay changes what is sent and how it is used, with one mechanism:

    read      H_i^0 = Enc(q, page_i)                       (every page node i)
    point     pi_i(t) proportional to sigma(s(h_t))         (evidence pointer: the
                                                            support head is the prior)
    emit      m_ik = sum_t softmax_t(<w_k, K h_t> + log pi_i(t)) h_t,   k = 1..M
    relay     H_i^{r+1} = Enc(q, [m_jk + rel(i, j)]_{j != i}, page_i)

The receiver's own encoder re-reads its page with the peers' pointed
evidence placed in its input, so every layer can bind a peer's fact to its
own tokens (a bridge entity, the other value of a comparison, a conflicting
date). The channel has a fixed bandwidth of M slots per page. The only side
information is one bit per pair (same agent or not); no page ids, ranks,
dates or condition labels enter the network.

``mode`` selects controlled variants that differ in exactly one factor:
  indep   no relay (independent readers; the decision pools the team only
          through the yes/no head, as in every variant);
  relay   Evidence Relay (pointer prior = support head);
  belief  same slots and re-reading, pointer prior = the agent's own answer
          distribution (what it believes) instead of its evidence;
  late    same evidence slots, fused after encoding by cross-attention
          (no re-reading);
  intra   relay restricted to pages of the same agent (no cross-agent channel).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from epinet.model.network import Block, loss_fn

MODES = ("indep", "relay", "belief", "late", "intra")


@dataclass
class RelayConfig:
    encoder: str = "microsoft/deberta-v3-base"
    mode: str = "relay"
    slots: int = 4
    rounds: int = 1
    heads: int = 8
    dropout: float = 0.1
    max_span: int = 30
    grad_ckpt: bool = False
    aux_weight: float = 0.5


class RelayNet(nn.Module):
    def __init__(self, cfg: RelayConfig, encoder: nn.Module | None = None):
        super().__init__()
        assert cfg.mode in MODES, cfg.mode
        self.cfg = cfg
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(cfg.encoder)
        self.encoder = encoder
        if cfg.grad_ckpt and hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        d = encoder.config.hidden_size
        self.d = d
        self.span = nn.Linear(d, 2)
        self.support = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.kind = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        if cfg.mode != "indep":
            self.queries = nn.Parameter(torch.randn(cfg.slots, d) * d ** -0.5)
            self.key = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
            self.write = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
            self.rel = nn.Embedding(2, d)
        if cfg.mode == "late":
            self.fuse = nn.ModuleList([Block(d, cfg.heads, cfg.dropout) for _ in range(2)])

    # ------------------------------------------------------------------ reading
    def read(self, ids, mask, msgs=None, msg_mask=None):
        if msgs is None:
            return self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        emb = self.encoder.embeddings.word_embeddings(ids)
        S = msgs.shape[1]
        x = torch.cat([emb[:, :1], msgs.to(emb.dtype), emb[:, 1:]], 1)
        m = torch.cat([mask[:, :1], msg_mask, mask[:, 1:]], 1)
        h = self.encoder(inputs_embeds=x, attention_mask=m.long()).last_hidden_state
        return torch.cat([h[:, :1], h[:, 1 + S:]], 1)

    def heads_out(self, X, valid, node_ok, B, n, sent):
        d = self.d
        logits = self.span(X).float()
        sup_tok = self.support(X).squeeze(-1).float()
        neg = -1e4
        start = logits[..., 0].masked_fill(~valid, neg).view(B, -1)
        end = logits[..., 1].masked_fill(~valid, neg).view(B, -1)
        cls = X[:, 0].view(B, n, d).float()
        w = node_ok.view(B, n, 1).float()
        pooled = (cls * w).sum(1) / w.sum(1).clamp(min=1.0)
        out = {"start": start, "end": end, "kind": self.kind(pooled.to(X.dtype)),
               "_logits": logits, "_sup": sup_tok}
        if sent is not None:
            flat_sent = sent.view(-1)
            keep = flat_sent >= 0
            n_sent = int(flat_sent.max().item()) + 1 if keep.any() else 0
            tok = sup_tok.reshape(-1)
            summed = torch.zeros(n_sent, device=X.device, dtype=tok.dtype)
            count = torch.zeros(n_sent, device=X.device, dtype=tok.dtype)
            summed.index_add_(0, flat_sent[keep], tok[keep])
            count.index_add_(0, flat_sent[keep], torch.ones_like(tok[keep]))
            out["support"] = summed / count.clamp(min=1)
        return out

    # ------------------------------------------------------------------ relay
    def emit(self, X, valid, prior):
        k = self.key(X)
        att = torch.einsum("md,nld->nml", self.queries.to(k.dtype), k).float() * self.d ** -0.5
        att = (att + prior.unsqueeze(1)).masked_fill(~valid.unsqueeze(1), -1e4)
        a = torch.softmax(att, -1).to(X.dtype)
        return torch.einsum("nml,nld->nmd", a, X)                         # [N, M, d]

    def route(self, slots, node_ok, agent_of, B, n, probe=None):
        """Every receiver i gets the slots of every sender j != i (same agent only for intra)."""
        M, d = slots.shape[1], self.d
        s = self.write(slots).view(B, n, M, d)
        if probe == "swap":                 # counterfactual: messages from another episode
            s = s.roll(1, dims=0)
        same = agent_of.view(n, 1) == agent_of.view(1, n)                 # [n, n]
        allow = ~torch.eye(n, dtype=torch.bool, device=slots.device)
        if self.cfg.mode == "intra":
            allow = allow & same
        ok = node_ok.roll(1, dims=0) if probe == "swap" else node_ok
        allow = allow.unsqueeze(0) & ok.view(B, 1, n)                     # [B, recv, send]
        if probe == "drop":
            allow = torch.zeros_like(allow)
        rel = self.rel(same.long().neg().add(1))                          # 0 same agent, 1 peer
        msgs = s.unsqueeze(1) + rel.view(1, n, n, 1, d).to(s.dtype)       # [B, recv, send, M, d]
        msgs = msgs.reshape(B * n, n * M, d)
        mmask = allow.unsqueeze(-1).expand(B, n, n, M).reshape(B * n, n * M)
        return msgs, mmask

    def forward(self, ids, mask, ctx, sent=None, probe=None):
        B, A, P, L = ids.shape
        n = A * P
        ids_f, mask_f = ids.view(B * n, L), mask.view(B * n, L)
        valid = ctx.view(B * n, L)
        node_ok = mask_f[:, 0]
        agent_of = torch.arange(A, device=ids.device).repeat_interleave(P)
        X = self.read(ids_f, mask_f)
        out0 = self.heads_out(X, valid, node_ok, B, n, sent)
        if self.cfg.mode == "indep" or n == 1:
            return {"final": out0, "rounds": [out0]}
        rounds = [out0]
        X0 = X
        cur = out0
        for _ in range(self.cfg.rounds):
            if self.cfg.mode == "belief":
                prior = torch.log_softmax(cur["_logits"][..., 0].masked_fill(~valid, -1e4), -1)
            else:
                prior = F.logsigmoid(cur["_sup"])
            slots = self.emit(X, valid, prior)
            msgs, mmask = self.route(slots, node_ok, agent_of, B, n, probe)
            if self.cfg.mode == "late":
                X = X0
                pad = ~mmask
                safe = pad.clone()
                safe[pad.all(1), 0] = False      # a receiver with no senders attends to a zero slot
                m_in = msgs.masked_fill(pad.unsqueeze(-1), 0.0)
                for blk in self.fuse:
                    X = blk(X, m_in, key_pad=safe)
            else:
                X = self.read(ids_f, mask_f, msgs, mmask)
            cur = self.heads_out(X, valid, node_ok, B, n, sent)
            rounds.append(cur)
        return {"final": cur, "rounds": rounds}


def relay_loss(out: dict, batch: dict, aux_weight: float = 0.5):
    total, parts = loss_fn(out["final"], batch, rate_weight=0.0)
    if len(out["rounds"]) > 1:
        aux, _ = loss_fn(out["rounds"][0], batch, rate_weight=0.0)
        total = total + aux_weight * aux
        parts["aux"] = float(aux)
    return total, parts
