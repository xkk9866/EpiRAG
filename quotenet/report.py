"""Tables, seed statistics and paired tests for the QuoteNet experiments.

    python -m quotenet.report experiments/quote [--json out.json]

Reads every ``<run>/test_rows*.jsonl``. Runs named ``<variant>_s<seed>`` are
grouped over seeds. Paired statistics compare two runs on the same
(dataset, qid, regime) episodes; the bootstrap resamples questions, so both
regimes of a question move together (10k draws, fixed seed).
"""

from __future__ import annotations

import collections
import glob
import json
import os
import random
import re
import statistics
import sys

DATA = ("hotpotqa", "2wiki", "musique")
REGIMES = ("split", "colocated")


def load(path):
    return {(r["dataset"], r["qid"], r["regime"]): r for r in map(json.loads, open(path, encoding="utf-8"))}


def runs(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "test_rows*.jsonl"))):
        name = os.path.basename(os.path.dirname(p)) + os.path.basename(p)[len("test_rows"):-len(".jsonl")]
        out[name] = load(p)
    return out


def em(rows, pred=lambda r: True, key="em"):
    v = [r[key] for r in rows.values() if pred(r)]
    return 100 * sum(v) / max(len(v), 1)


def paired(a, b, pred=lambda r: True, draws=10000, seed=0):
    keys = [k for k in a if k in b and pred(a[k])]
    byq = collections.defaultdict(list)
    for k in keys:
        byq[(k[0], k[1])].append(a[k]["em"] - b[k]["em"])
    qs = list(byq)
    if not qs:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    tot = lambda s: sum(sum(byq[q]) for q in s) / max(sum(len(byq[q]) for q in s), 1)
    obs = tot(qs)
    rng = random.Random(seed)
    boots = sorted(tot([qs[rng.randrange(len(qs))] for _ in qs]) for _ in range(draws))
    lo, hi = boots[int(0.025 * draws)], boots[int(0.975 * draws)]
    p = 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / draws
    return 100 * obs, (100 * lo, 100 * hi), min(p, 1.0)


def group(R):
    g = collections.defaultdict(dict)
    for name, rows in R.items():
        m = re.match(r"(.+)_s(\d+)$", name)
        base, seed = (m.group(1), int(m.group(2))) if m else (name, 0)
        g[base][seed] = rows
    return g


def cells(rows):
    out = {}
    for d in DATA:
        for r in REGIMES:
            out[f"{d}/{r}"] = em(rows, lambda x, d=d, r=r: x["dataset"] == d and x["regime"] == r)
    for r in REGIMES:
        out[f"avg/{r}"] = sum(out[f"{d}/{r}"] for d in DATA) / len(DATA)
    return out


def main(root, out_json=None):
    R = runs(root)
    G = group(R)
    report = {"cells": {}, "qtype": {}, "paired": {}, "quotes": {}}
    print("EM (mean ± sd over seeds) by dataset / regime")
    cols = [f"{d}/{r}" for r in REGIMES for d in DATA] + [f"avg/{r}" for r in REGIMES]
    print("variant".ljust(24) + "".join(c.replace("hotpotqa", "hotpot").rjust(17) for c in cols))
    for base, seeds in sorted(G.items()):
        per = [cells(rows) for rows in seeds.values()]
        line = base.ljust(24)
        report["cells"][base] = {}
        for c in cols:
            v = [p[c] for p in per]
            m, s = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
            report["cells"][base][c] = (m, s, len(v))
            line += (f"{m:.1f}±{s:.1f}({len(v)})" if len(v) > 1 else f"{m:.1f}").rjust(17)
        print(line)

    print("\nsplit-regime EM by question type (seed-averaged)")
    qtypes = sorted({(r["dataset"], r["qtype"]) for rows in R.values() for r in rows.values()})
    print("variant".ljust(24) + "".join(f"{d[:3]}:{t[:8]}".rjust(14) for d, t in qtypes))
    for base, seeds in sorted(G.items()):
        line = base.ljust(24)
        report["qtype"][base] = {}
        for d, t in qtypes:
            v = statistics.mean(em(rows, lambda x, d=d, t=t: x["dataset"] == d and x["qtype"] == t and x["regime"] == "split")
                                for rows in seeds.values())
            report["qtype"][base][f"{d}/{t}"] = v
            line += f"{v:14.1f}"
        print(line)

    REF = "ic" if "ic" in G else "indep"
    if REF in G:
        print(f"\npaired vs {REF}, same seed (dEM, 95% CI, p), split regime")
        for base, seeds in sorted(G.items()):
            if base in (REF, "base"):
                continue
            for seed, rows in sorted(seeds.items()):
                ref = G[REF].get(seed) or next(iter(G[REF].values()))
                for d in DATA + ("ALL",):
                    for r in REGIMES:
                        pr = lambda x, d=d, r=r: (d == "ALL" or x["dataset"] == d) and x["regime"] == r
                        obs, ci, p = paired(rows, ref, pr, draws=2000)
                        report["paired"][f"{base}/s{seed}/{d}/{r}"] = (obs, ci, p)
                        if d == "ALL" or r == "split":
                            print(f"  {base:20s} s{seed:<3d} {d:9s} {r:9s} {obs:+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] p={p:.4f}")

    print("\nwhat was quoted (split regime): share of supporting-fact tokens, share of quotes naming the answer")
    for name, rows in sorted(R.items()):
        sup, ans, n = 0.0, 0.0, 0
        gsup, nsup = 0.0, 0
        for r in rows.values():
            if r["regime"] != "split" or not r.get("quotes"):
                continue
            for a, q in enumerate(r["quotes"]):
                sup += q["support"]; ans += q["answer"]; n += 1
                if a in r["gold_agents"]:
                    gsup += q["support"]; nsup += 1
        if n:
            report["quotes"][name] = {"support": sup / n, "support_gold_agents": gsup / max(nsup, 1), "answer": ans / n}
            print(f"  {name:28s} support {100*sup/n:5.1f}%  (gold-holding agents {100*gsup/max(nsup,1):5.1f}%)  names answer {100*ans/n:5.1f}%")
    if out_json:
        json.dump(report, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out in args:
        args.remove(out)
    main(args[0] if args else "experiments/quote", out)
