"""Correctness checks on a tiny random encoder (not an experiment)."""

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from epinet.data.encode import EncodeConfig, EpisodeDataset, collate
from epinet.data.examples import sample_examples
from epinet.data.views import CONDITIONS
from epinet.model.network import EpiNet, NetConfig, decode, loss_fn


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    exs = sample_examples("2wiki", "validation", 3, seed=7)
    ds = EpisodeDataset(tok, exs, EncodeConfig(max_len=128), fixed=list(CONDITIONS))
    batch = collate([ds[i] for i in range(len(ds))])
    for i in range(len(ds)):
        it = ds[i]
        print(it["meta"]["condition"], it["meta"]["answers"], "gold starts", int(it["start"].sum()), "kind", it["kind"])
    enc = BertModel(BertConfig(vocab_size=tok.vocab_size + 200, hidden_size=64, num_hidden_layers=2,
                               num_attention_heads=4, intermediate_size=128, max_position_embeddings=256))
    model = EpiNet(NetConfig(heads=4, dropout=0.0, grad_ckpt=False), encoder=enc)
    out = model(batch["ids"], batch["mask"], batch["ctx"], batch["sent"])
    loss, parts = loss_fn(out, batch)
    loss.backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad and "pooler" not in n)
    print("loss", float(loss), parts)
    print(decode(out, batch)[:3])

    # agent permutation equivariance: reordering views must not change the answer distribution
    model.eval()
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        o1 = model(batch["ids"], batch["mask"], batch["ctx"])
        o2 = model(batch["ids"][:, perm], batch["mask"][:, perm], batch["ctx"][:, perm])
    B = batch["ids"].shape[0]
    s1 = o1["start"].view(B, 3, -1)[:, perm]
    s2 = o2["start"].view(B, 3, -1)
    assert torch.allclose(s1, s2, atol=1e-4), (s1 - s2).abs().max()
    assert torch.allclose(o1["kind"], o2["kind"], atol=1e-4)
    print("permutation equivariance ok")

    # INT8 checks: quantized modules exist, and weight codes are integers in [-127, 127]
    from epinet.model.quant import QAttention, QuantLinear, fake_quant, tensor_scale
    n_lin = sum(isinstance(m, QuantLinear) for m in model.modules())
    n_att = sum(isinstance(m, QAttention) for m in model.modules())
    assert n_lin > 0 and n_att >= 1, (n_lin, n_att)
    assert not any(isinstance(m, torch.nn.MultiheadAttention) for m in model.modules())
    w = model.bridge.attn.q_proj.weight.detach()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    codes = torch.round(w / scale)
    assert torch.equal(codes, codes.round()) and codes.abs().max() <= 127
    x = torch.randn(64, 64)
    levels = torch.unique(torch.round(fake_quant(x, tensor_scale(x)) / tensor_scale(x)))
    assert levels.numel() <= 255
    print("int8 modules", n_lin, "linear", n_att, "attention", "codes ok")


if __name__ == "__main__":
    main()
