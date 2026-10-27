"""Data and gradient checks for QuoteNet (not an experiment)."""

import collections

import torch
from transformers import AutoTokenizer

from epinet.data.examples import sample_examples
from quotenet.encode import Episodes, collate, encode_episode
from quotenet.model import QuoteConfig, QuoteNet, decode, quote_loss
from quotenet.views import REGIMES, build_views

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
# 1. views: no page appears twice, supporting pages present, regimes as specified
stats = collections.Counter()
for d in ("hotpotqa", "2wiki", "musique"):
    exs = sample_examples(d, "validation", 60, seed=7)
    for ex in exs:
        for r in REGIMES:
            v = build_views(ex, r)
            pids = [p.pid for view in v for p in view]
            assert len(pids) == len(set(pids)), (ex.qid, r)
            holders = [a for a, view in enumerate(v) if any(p.gold for p in view)]
            ng = sum(p.gold for view in v for p in view)
            assert ng == sum(p.gold for p in ex.pages) or ng >= 1
            stats[(d, r, len(holders))] += 1
            # position of gold pages inside a view should be uniform
            for view in v:
                for i, p in enumerate(view):
                    if p.gold:
                        stats[("pos", i)] += 1
            enc = encode_episode(tok, ex, r)
            n_start = sum(sum(ag["start"]) for ag in enc["agents"])
            stats[(d, "has_start")] += n_start > 0
            stats[(d, "len")] = max(stats[(d, "len")], max(len(ag["ids"]) for ag in enc["agents"]))
            stats[(d, "sup")] += sum(enc["sent_label"])
print(sorted(stats.items(), key=str))

# 2. model: shapes, gradients reach the pointer through the receivers, decode works
exs = sample_examples("2wiki", "validation", 4, seed=7)
ds = Episodes(tok, exs, fixed=list(REGIMES))
batch = collate([ds[i] for i in range(4)], tok.pad_token_id)
dev = "cuda:0"
t = {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}
for mode in ("indep", "quote", "belief", "slots", "sg", "free"):
    torch.manual_seed(0)
    m = QuoteNet(QuoteConfig(encoder="microsoft/deberta-v3-base", mode=mode, k=16), sep_id=tok.sep_token_id,
                 null_id=tok.mask_token_id).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(t["ids"], t["mask"], t["ctx"], t["sent"])
    loss, parts = quote_loss(out, t)
    loss.backward()
    g_sup = m.support[1].weight.grad.abs().sum().item()
    print(mode, round(float(loss), 3), parts, "support-head grad", g_sup,
          "null grad", None if not hasattr(m, "null") or m.null.grad is None else m.null.grad.abs().sum().item())
    if mode == "quote":
        # receiver gradient alone (no support loss) must reach the pointer
        m.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(t["ids"], t["mask"], t["ctx"], None)
        from epinet.model.network import loss_fn
        l2, _ = loss_fn(out["final"], t, rate_weight=0.0)
        l2.backward()
        print("  pointer grad from receivers only:", m.support[1].weight.grad.abs().sum().item())
        idx, valid = out["quotes"][0]
        print("  quote:", tok.decode([batch["views"][0][0]["ids"][j] for j in idx[0, 0][valid[0, 0]].tolist()]))
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(t["ids"], t["mask"], t["ctx"])
    print("  decode", [p["answer"][:30] for p in decode(out["final"], {**batch, "page": t["page"]})])
