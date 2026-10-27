"""Train and evaluate Evidence Relay and its controlled variants.

    python -m epinet.train_relay --name relay --mode relay
    python -m epinet.train_relay --name indep --mode indep
    python -m epinet.train_relay --name relay --mode relay --eval-only --probe swap
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from epinet.data.encode import EncodeConfig, EpisodeDataset, collate
from epinet.data.examples import sample_examples
from epinet.data.views import CONDITIONS
from epinet.model.network import decode
from epinet.model.relay import MODES, RelayConfig, RelayNet, relay_loss
from epinet.text import best_em, best_f1


def args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="relay")
    ap.add_argument("--mode", default="relay", choices=MODES)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--datasets", default="hotpotqa,2wiki,musique")
    ap.add_argument("--n-train", type=int, default=2000, help="per dataset")
    ap.add_argument("--n-dev", type=int, default=150, help="per dataset")
    ap.add_argument("--n-test", type=int, default=300, help="per dataset")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--centralize", action="store_true")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="experiments/relay")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--probe", default=None, choices=[None, "swap", "drop"])
    return ap.parse_args()


def evaluate(model, tok, examples, ecfg, device, conditions=CONDITIONS, batch=8, probe=None, workers=2):
    ds = EpisodeDataset(tok, examples, ecfg, fixed=list(conditions))
    dl = DataLoader(ds, batch_size=batch, collate_fn=collate, num_workers=workers)
    rows = []
    model.eval()
    with torch.no_grad():
        for b in dl:
            t = {k: v.to(device) for k, v in b.items() if torch.is_tensor(v)}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(t["ids"], t["mask"], t["ctx"], probe=probe)["final"]
            t["meta"] = b["meta"]
            for meta, pred in zip(b["meta"], decode(out, t, model.cfg.max_span)):
                rows.append({"agent": pred["agent"], "page": pred["page"],
                    "qid": meta["qid"], "dataset": meta["dataset"], "qtype": meta["qtype"],
                    "condition": meta["condition"], "answer": pred["answer"], "gold": meta["answers"],
                    "em": best_em(pred["answer"], meta["answers"]), "f1": best_f1(pred["answer"], meta["answers"]),
                })
    model.train()
    return rows


def summary(rows):
    agg = {}
    for r in rows:
        for key in ((r["dataset"], r["condition"]), (r["dataset"], "ALL")):
            agg.setdefault(key, []).append((r["em"], r["f1"]))
    return {f"{k[0]}/{k[1]}": [round(100 * sum(e for e, _ in v) / len(v), 2), round(100 * sum(f for _, f in v) / len(v), 2), len(v)]
            for k, v in sorted(agg.items())}


def main():
    a = args_()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.encoder)
    ecfg = EncodeConfig(max_len=a.max_len, centralize=a.centralize, pages_per_agent=9 if a.centralize else 3)
    rcfg = RelayConfig(encoder=a.encoder, mode=a.mode, slots=a.slots, rounds=a.rounds,
                       grad_ckpt=a.ckpt, aux_weight=a.aux_weight)
    model = RelayNet(rcfg).to(a.device)
    run_dir = os.path.join(a.out, a.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "args.json"), "w"), indent=1)
    names = a.datasets.split(",")
    dev = {d: sample_examples(d, "validation", a.n_dev, seed=7) for d in names}
    test = {d: sample_examples(d, "validation", a.n_test, seed=7, offset=a.n_dev) for d in names}
    ckpt = os.path.join(run_dir, "model.pt")

    if not a.eval_only:
        train = []
        for d in names:
            train += sample_examples(d, "train", a.n_train, seed=a.seed)
        ds = EpisodeDataset(tok, train, ecfg, seed=a.seed)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=collate,
                        num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
        enc_ids = {id(p) for p in model.encoder.parameters()}
        heads = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": a.lr},
                                 {"params": heads, "lr": a.head_lr}], weight_decay=0.01)
        total = a.epochs * len(dl) // a.accum
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=max(total, 1), pct_start=0.06)
        best, step, t0 = -1.0, 0, time.time()
        for epoch in range(a.epochs):
            for i, b in enumerate(dl):
                t = {k: v.to(a.device) for k, v in b.items() if torch.is_tensor(v)}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(t["ids"], t["mask"], t["ctx"], t["sent"])
                loss, parts = relay_loss(out, t, a.aux_weight)
                (loss / a.accum).backward()
                if (i + 1) % a.accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 50 == 0:
                        print(f"ep{epoch} step{step}/{total} loss {float(loss):.3f} {parts} {time.time()-t0:.0f}s", flush=True)
            rows = []
            for d in names:
                rows += evaluate(model, tok, dev[d], ecfg, a.device, workers=a.workers)
            s = summary(rows)
            score = sum(v[0] for k, v in s.items() if k.endswith("/ALL")) / len(names)
            print("dev", json.dumps(s), flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=a.device))
    rows = []
    for d in names:
        rows += evaluate(model, tok, test[d], ecfg, a.device, probe=a.probe, workers=a.workers)
    tag = f"_{a.probe}" if a.probe else ""
    with open(os.path.join(run_dir, f"test_rows{tag}.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summary(rows)
    json.dump(s, open(os.path.join(run_dir, f"test_summary{tag}.json"), "w"), indent=1)
    print("test", json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
