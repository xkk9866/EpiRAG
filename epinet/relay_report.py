"""Tables and paired tests for the Evidence Relay experiments.

    python -m epinet.relay_report experiments/relay

Reads every ``<run>/test_rows*.jsonl``. Paired statistics use the same
(qid, condition) episodes; the bootstrap resamples questions (all six
conditions of a question move together), 10k draws, fixed seed.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import random
import sys

CONDS = ("clean", "asymmetry", "corruption", "temporal", "incomplete", "collusion")
DATA = ("hotpotqa", "2wiki", "musique")
SPLIT_TYPES = {("2wiki", "comparison"), ("2wiki", "bridge_comparison"), ("hotpotqa", "comparison")}


def load(path):
    return {(r["dataset"], r["qid"], r["condition"]): r for r in map(json.loads, open(path, encoding="utf-8"))}


def runs(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "test_rows*.jsonl"))):
        name = os.path.basename(os.path.dirname(p)) + os.path.basename(p)[len("test_rows"):-len(".jsonl")]
        out[name] = load(p)
    return out


def mean(rows, key="em", pred=lambda r: True):
    v = [r[key] for r in rows.values() if pred(r)]
    return 100 * sum(v) / max(len(v), 1), len(v)


def paired(a, b, pred=lambda r: True, draws=10000, seed=0):
    keys = [k for k in a if k in b and pred(a[k])]
    byq = collections.defaultdict(list)
    for k in keys:
        byq[(k[0], k[1])].append(a[k]["em"] - b[k]["em"])
    qs = list(byq)
    if not qs:
        return float("nan"), float("nan"), float("nan")
    tot = lambda sample: sum(sum(byq[q]) for q in sample) / max(sum(len(byq[q]) for q in sample), 1)
    obs = tot(qs)
    rng = random.Random(seed)
    boots = sorted(tot([qs[rng.randrange(len(qs))] for _ in qs]) for _ in range(draws))
    lo, hi = boots[int(0.025 * draws)], boots[int(0.975 * draws)]
    p = 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / draws
    return 100 * obs, (100 * lo, 100 * hi), min(p, 1.0)


def main(root):
    R = runs(root)
    print("runs:", ", ".join(f"{k}({len(v)})" for k, v in R.items()))
    print("\nEM by dataset/condition")
    hdr = "run".ljust(22) + "".join(f"{d[:5]}/{c[:5]}".rjust(12) for d in DATA for c in ("ALL",) + CONDS)
    print(hdr)
    for name, rows in R.items():
        cells = []
        for d in DATA:
            for c in ("ALL",) + CONDS:
                m, _ = mean(rows, pred=lambda r, d=d, c=c: r["dataset"] == d and (c == "ALL" or r["condition"] == c))
                cells.append(f"{m:12.1f}")
        print(name.ljust(22) + "".join(cells))
    print("\nsplit-evidence comparison items (asymmetry/corruption/temporal) vs others")
    split = lambda r: (r["dataset"], r["qtype"]) in SPLIT_TYPES and r["condition"] in ("asymmetry", "corruption", "temporal")
    for name, rows in R.items():
        print(name.ljust(22), "split %.1f (n=%d)" % mean(rows, pred=split), " rest %.1f" % mean(rows, pred=lambda r: not split(r))[0])
    base = next((k for k in R if k.startswith("indep")), None)
    if base:
        print(f"\npaired vs {base}: dEM, 95% CI, p")
        for name, rows in R.items():
            if name == base:
                continue
            for d in DATA + ("ALL",):
                pr = (lambda r, d=d: d == "ALL" or r["dataset"] == d)
                obs, ci, p = paired(rows, R[base], pr, draws=2000)
                print(f"  {name:20s} {d:9s} {obs:+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] p={p:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/relay")
