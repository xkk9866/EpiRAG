"""Train and evaluate QuoteNet and its controlled variants.

    python -m quotenet.train --name quote_k32_s13 --mode quote --k 32
    python -m quotenet.train --name indep_s13 --mode indep
    python -m quotenet.train --name central_s13 --mode indep --centralize
    python -m quotenet.train --name quote_k32_s13 --mode quote --eval-only --probe swap
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from epinet.data.examples import sample_examples as _real
from quotenet.synth import sample as _synth


def sample_examples(d, split, n, seed, offset=0):
    return (_synth if d.startswith("synth") else _real)(d, split, n, seed, offset)
from epinet.text import best_em, best_f1, normalize_answer
from quotenet.encode import Episodes, collate
from quotenet.model import MODES, QuoteConfig, QuoteNet, decode, quote_loss
from quotenet.views import REGIMES


def args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="quote")
    ap.add_argument("--mode", default="quote", choices=MODES)
    ap.add_argument("--k", type=int, default=32, help="quoted tokens per agent and round")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--datasets", default="hotpotqa,2wiki,musique")
    ap.add_argument("--n-train", type=int, default=3000, help="per dataset")
    ap.add_argument("--n-dev", type=int, default=150, help="per dataset")
    ap.add_argument("--n-test", type=int, default=500, help="per dataset")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--centralize", action="store_true")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="experiments/quote")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--probe", default=None, choices=[None, "swap", "drop", "random"])
    ap.add_argument("--init", default=None, help="stage-1 checkpoint (strict=False)")
    ap.add_argument("--train-seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke runs only")
    return ap.parse_args()


def quote_stats(batch, out, tok, b):
    """What each agent disclosed: text, share of supporting-fact tokens, answer mention."""
    if not out["quotes"] or out["quotes"][-1] is None:
        return None
    idx, valid = out["quotes"][-1]
    stats = []
    labels = batch["sent_label"]
    for a in range(idx.shape[1]):
        sel = idx[b, a][valid[b, a]].tolist()
        ids = [batch["views"][b][a]["ids"][t] for t in sel]
        sids = batch["sent"][b, a, sel].tolist()
        sup = [float(labels[s]) if s >= 0 else 0.0 for s in sids]
        text = tok.decode(ids)
        ans = any(normalize_answer(g) and f" {normalize_answer(g)} " in f" {normalize_answer(text)} "
                  for g in batch["meta"][b]["answers"] if normalize_answer(g) not in {"yes", "no"})
        stats.append({"text": text, "support": sum(sup) / max(len(sup), 1), "answer": ans})
    return stats


def evaluate(model, tok, examples, device, max_len, centralize, batch=4, probe=None, workers=2, keep_quotes=False):
    ds = Episodes(tok, examples, max_len, centralize, fixed=list(REGIMES))
    dl = DataLoader(ds, batch_size=batch, collate_fn=lambda x: collate(x, tok.pad_token_id), num_workers=workers)
    rows = []
    model.eval()
    with torch.no_grad():
        for bt in dl:
            t = {k: v.to(device) for k, v in bt.items() if torch.is_tensor(v)}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(t["ids"], t["mask"], t["ctx"], probe=probe, qok=t["qok"])
            preds = decode(out["final"], {**bt, "page": t["page"]}, model.cfg.max_span)
            for b, (meta, pred) in enumerate(zip(bt["meta"], preds)):
                row = {k: meta[k] for k in ("qid", "dataset", "qtype", "regime", "gold_agents")}
                row.update({"answer": pred["answer"], "agent": pred["agent"], "gold": meta["answers"],
                            "em": best_em(pred["answer"], meta["answers"]),
                            "f1": best_f1(pred["answer"], meta["answers"])})
                if keep_quotes:
                    row["quotes"] = quote_stats(bt, out, tok, b)
                rows.append(row)
    model.train()
    return rows


def summary(rows):
    agg = {}
    for r in rows:
        for key in ((r["dataset"], r["regime"]), (r["dataset"], "ALL")):
            agg.setdefault(key, []).append((r["em"], r["f1"]))
    return {f"{k[0]}/{k[1]}": [round(100 * sum(e for e, _ in v) / len(v), 2),
                               round(100 * sum(f for _, f in v) / len(v), 2), len(v)]
            for k, v in sorted(agg.items())}


def main():
    a = args_()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.encoder)
    cfg = QuoteConfig(encoder=a.encoder, mode=a.mode, k=a.k, rounds=a.rounds, aux_weight=a.aux_weight)
    model = QuoteNet(cfg, sep_id=tok.sep_token_id, null_id=tok.mask_token_id).to(a.device)
    run_dir = os.path.join(a.out, a.name)
    os.makedirs(run_dir, exist_ok=True)
    names = a.datasets.split(",")
    dev = {d: sample_examples(d, "validation", a.n_dev, seed=7) for d in names}
    test = {d: sample_examples(d, "validation", a.n_test, seed=7, offset=a.n_dev) for d in names}
    ckpt = os.path.join(run_dir, "model.pt")
    if a.init:
        print("init", a.init, model.load_state_dict(torch.load(a.init, map_location=a.device), strict=False), flush=True)
    pad = tok.pad_token_id

    if not a.eval_only:
        json.dump(vars(a), open(os.path.join(run_dir, "args.json"), "w"), indent=1)
        train = []
        for d in names:
            train += sample_examples(d, "train", a.n_train, seed=a.train_seed if a.train_seed is not None else a.seed)
        ds = Episodes(tok, train, a.max_len, a.centralize, seed=a.seed)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=lambda x: collate(x, pad),
                        num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
        enc_ids = {id(p) for p in model.encoder.parameters()}
        heads = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": a.lr},
                                 {"params": heads, "lr": a.head_lr}], weight_decay=0.01)
        total = a.epochs * len(dl) // a.accum
        if a.max_steps:
            total = min(total, a.max_steps)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=max(total, 1),
                                                    pct_start=0.06)
        best, step, t0 = -1.0, 0, time.time()
        for epoch in range(a.epochs):
            for i, bt in enumerate(dl):
                t = {k: v.to(a.device) for k, v in bt.items() if torch.is_tensor(v)}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(t["ids"], t["mask"], t["ctx"], t["sent"], qok=t["qok"])
                loss, parts = quote_loss(out, t, a.aux_weight)
                (loss / a.accum).backward()
                if (i + 1) % a.accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 50 == 0:
                        parts = {k: round(v, 3) for k, v in parts.items()}
                        print(f"ep{epoch} step{step}/{total} loss {float(loss):.3f} {parts} {time.time()-t0:.0f}s", flush=True)
                    if step >= total:
                        break
            rows = []
            for d in names:
                rows += evaluate(model, tok, dev[d], a.device, a.max_len, a.centralize, workers=a.workers)
            s = summary(rows)
            score = sum(v[0] for k, v in s.items() if k.endswith("/ALL")) / len(names)
            print("dev", json.dumps(s), flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), ckpt)
            if step >= total:
                break
    model.load_state_dict(torch.load(ckpt, map_location=a.device))
    rows = []
    for d in names:
        rows += evaluate(model, tok, test[d], a.device, a.max_len, a.centralize, probe=a.probe,
                         workers=a.workers, keep_quotes=a.mode not in ("indep", "reread"))
    tag = f"_{a.probe}" if a.probe else ""
    with open(os.path.join(run_dir, f"test_rows{tag}.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summary(rows)
    json.dump(s, open(os.path.join(run_dir, f"test_summary{tag}.json"), "w"), indent=1)
    print("test", json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
