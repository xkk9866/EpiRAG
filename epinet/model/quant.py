"""INT8 quantization-aware training for EpiNet.

Every linear map and every attention matrix product runs on INT8 codes in
the forward pass:

* weights: symmetric, per output channel, codes in [-127, 127];
* activations entering a linear map or a matrix product: symmetric, per
  tensor, scale from the current tensor's max (dynamic), codes in [-127, 127];
* attention probabilities feeding AV: symmetric per tensor, dynamic.

Gradients pass straight through the rounding (straight-through estimator),
so the model trains against its own INT8 forward pass.

Not quantized: softmax, LayerNorm, GELU, residual additions, and the loss.
These stay in floating point, as in standard INT8 QAT (Jacob et al., 2018).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX = 127.0


def fake_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round to INT8 codes and dequantize; gradient is the identity."""
    codes = torch.clamp(torch.round(x / scale), -QMAX, QMAX)
    return x + (codes * scale - x).detach()


def tensor_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().abs().amax().clamp(min=1e-8) / QMAX).to(x.dtype)


class QuantLinear(nn.Linear):
    def forward(self, x):
        w_scale = (self.weight.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / QMAX)
        w_q = fake_quant(self.weight, w_scale)
        x_q = fake_quant(x, tensor_scale(x))
        return F.linear(x_q, w_q, self.bias)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "QuantLinear":
        q = cls(lin.in_features, lin.out_features, bias=lin.bias is not None)
        q.weight.data.copy_(lin.weight.data)
        if lin.bias is not None:
            q.bias.data.copy_(lin.bias.data)
        return q.to(lin.weight.device)


class QAttention(nn.Module):
    """Multi-head attention with INT8 projections and INT8 score/value products.

    Mirrors the subset of nn.MultiheadAttention that EpiNet uses. The
    ``attn_mask`` convention is the same: a float additive mask of shape
    (batch*heads, L, S), and ``key_padding_mask`` is True where a key is ignored.
    """

    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.d, self.heads, self.dh = d, heads, d // heads
        self.q_proj = QuantLinear(d, d)
        self.k_proj = QuantLinear(d, d)
        self.v_proj = QuantLinear(d, d)
        self.out_proj = QuantLinear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False):
        N, L, _ = query.shape
        S = key.shape[1]
        H, dh = self.heads, self.dh
        q = self.q_proj(query).view(N, L, H, dh).transpose(1, 2)
        k = self.k_proj(key).view(N, S, H, dh).transpose(1, 2)
        v = self.v_proj(value).view(N, S, H, dh).transpose(1, 2)
        # INT8 QK^T: codes of q and k, dequantized product
        scores = fake_quant(q, tensor_scale(q)) @ fake_quant(k, tensor_scale(k)).transpose(-1, -2)
        scores = scores / math.sqrt(dh)
        if attn_mask is not None:
            scores = scores + attn_mask.view(N, H, L, S).to(scores.dtype)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(N, 1, 1, S), float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        probs = self.drop(probs)
        # INT8 AV
        out = fake_quant(probs, tensor_scale(probs)) @ fake_quant(v, tensor_scale(v))
        out = out.transpose(1, 2).reshape(N, L, self.d)
        return self.out_proj(out), None

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention) -> "QAttention":
        d = mha.embed_dim
        q = cls(d, mha.num_heads, mha.dropout)
        w_q, w_k, w_v = mha.in_proj_weight.data.chunk(3, dim=0)
        q.q_proj.weight.data.copy_(w_q)
        q.k_proj.weight.data.copy_(w_k)
        q.v_proj.weight.data.copy_(w_v)
        if mha.in_proj_bias is not None:
            b_q, b_k, b_v = mha.in_proj_bias.data.chunk(3, dim=0)
            q.q_proj.bias.data.copy_(b_q)
            q.k_proj.bias.data.copy_(b_k)
            q.v_proj.bias.data.copy_(b_v)
        q.out_proj.weight.data.copy_(mha.out_proj.weight.data)
        if mha.out_proj.bias is not None:
            q.out_proj.bias.data.copy_(mha.out_proj.bias.data)
        return q.to(mha.in_proj_weight.device)


def quantize_module(module: nn.Module) -> nn.Module:
    """Replace every nn.MultiheadAttention and nn.Linear below ``module`` in place."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, QAttention.from_mha(child))
        elif isinstance(child, QuantLinear):
            continue
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear.from_linear(child))
        else:
            quantize_module(child)
    return module


def int8_codes(model: nn.Module) -> dict[str, float]:
    """Diagnostic: fraction of weight codes that saturate at +-127 per layer type."""
    stats = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            s = mod.weight.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / QMAX
            codes = torch.round(mod.weight.detach() / s).clamp(-QMAX, QMAX)
            stats[name] = float((codes.abs() == QMAX).float().mean())
    return stats
