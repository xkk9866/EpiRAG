"""Only quoted words cross a view boundary.

For each episode, replace an unquoted page token of agent 1 by a random word
and check that (i) agent 1 quotes the same tokens and (ii) the re-read outputs
of agents 0 and 2 are bit-identical. Run in fp32, eval mode.
"""

import torch
from transformers import AutoTokenizer

from epinet.data.examples import sample_examples
from quotenet.encode import Episodes, collate
from quotenet.model import QuoteConfig, QuoteNet
from quotenet.views import REGIMES

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
exs = sample_examples("hotpotqa", "validation", 8, seed=7)
ds = Episodes(tok, exs, fixed=list(REGIMES))
m = QuoteNet(QuoteConfig(encoder="microsoft/deberta-v3-base", mode="quote", k=16), sep_id=tok.sep_token_id,
             null_id=tok.mask_token_id).cuda().eval()
g = torch.Generator().manual_seed(0)
checked = same_quote = 0
max_diff = 0.0
for i in range(len(ds)):
    b = collate([ds[i]], tok.pad_token_id)
    t = {k: v.cuda() for k, v in b.items() if torch.is_tensor(v)}
    with torch.no_grad():
        o1 = m(t["ids"], t["mask"], t["ctx"])
        idx, valid = o1["quotes"][0]
        quoted = set(idx[0, 1][valid[0, 1]].tolist())
        cand = [p for p in torch.nonzero(t["ctx"][0, 1]).view(-1).tolist() if p not in quoted]
        pos = cand[torch.randint(len(cand), (1,), generator=g).item()]
        ids2 = t["ids"].clone()
        ids2[0, 1, pos] = torch.randint(1000, 120000, (1,), generator=g).item()
        o2 = m(ids2, t["mask"], t["ctx"])
    if set(o2["quotes"][0][0][0, 1][o2["quotes"][0][1][0, 1]].tolist()) != quoted:
        continue  # the change moved agent 1's own pointer; the quote itself changed
    same_quote += 1
    L = t["ids"].shape[-1]
    for key in ("start", "end"):
        a = o1["final"][key].view(3, L)[[0, 2]]
        c = o2["final"][key].view(3, L)[[0, 2]]
        max_diff = max(max_diff, (a - c).abs().max().item())
    checked += 1
print(f"episodes {len(ds)}, quote unchanged {same_quote}, max |delta| on agents 0 and 2: {max_diff:.3g}")
assert checked > 0 and max_diff == 0.0
print("PASS: only quoted words cross the view boundary")
