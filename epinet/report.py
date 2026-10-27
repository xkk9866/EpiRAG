"""Main/ablation tables and learned-weight analysis for EpiNet v2.

    python -m epinet.report --v2 experiments/v2 --llm ~/epirag/experiments/llm_small/Qwen2.5-7B-Instruct@int8.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict

from epinet.data.views import CONDITIONS

DS = ("hotpotqa", "2wiki", "musique")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def cells(rows):
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["condition"])].append((r["em"], r["f1"]))
        agg[(r["dataset"], "ALL")].append((r["em"], r["f1"]))
    return {k: (100 * sum(e for e, _ in v) / len(v), 100 * sum(f for _, f in v) / len(v)) for k, v in agg.items()}


def paired_boot(a, b, n=2000, seed=0):
    """Question-clustered bootstrap p-value that mean EM(a) > EM(b)."""
    ka = defaultdict(list); kb = defaultdict(list)
    for r in a: ka[(r["dataset"], r["qid"])].append(r["em"])
    for r in b: kb[(r["dataset"], r["qid"])].append(r["em"])
    keys = [k for k in ka if k in kb]
    d = [sum(ka[k]) / len(ka[k]) - sum(kb[k]) / len(kb[k]) for k in keys]
    rng = random.Random(seed)
    obs = sum(d) / len(d)
    worse = 0
    for _ in range(n):
        s = sum(d[rng.randrange(len(d))] for _ in d) / len(d)
        worse += s <= 0
    return obs * 100, worse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="experiments/v2")
    ap.add_argument("--llm", default="")
    ap.add_argument("--out", default="experiments/v2/report.md")
    a = ap.parse_args()
    runs = {os.path.basename(os.path.dirname(p)): load(p) for p in glob.glob(os.path.join(a.v2, "*", "test_rows.jsonl"))}
    if a.llm and os.path.exists(a.llm):
        for r in load(a.llm):
            runs.setdefault("llm_" + r["method"], []).append(r)
    order = [m for m in ("llm_vote", "llm_pool", "llm_debate", "llm_madam", "central", "nochannel", "noindep",
                         "nocorrob", "nowrite", "epinet2") if m in runs]
    C = {m: cells(runs[m]) for m in order}
    lines = ["# EpiNet v2 results (test, EM / F1)", ""]
    for d in DS:
        lines += [f"## {d}", "", "| method | " + " | ".join(CONDITIONS) + " | stress-avg | ALL |",
                  "|---" * (len(CONDITIONS) + 3) + "|"]
        for m in order:
            c = C[m]
            if (d, "ALL") not in c:
                continue
            stress = [c[(d, k)][0] for k in CONDITIONS if k != "clean" and (d, k) in c]
            row = [f"{c[(d, k)][0]:.1f}/{c[(d, k)][1]:.1f}" if (d, k) in c else "-" for k in CONDITIONS]
            lines.append(f"| {m} | " + " | ".join(row) + f" | {sum(stress)/len(stress):.1f} | {c[(d,'ALL')][0]:.1f}/{c[(d,'ALL')][1]:.1f} |")
        lines.append("")
    if "epinet2" in runs:
        lines += ["## Paired bootstrap (EM, epinet2 minus X, question-clustered)", ""]
        for m in order:
            if m != "epinet2":
                diff, p = paired_boot(runs["epinet2"], runs[m])
                lines.append(f"- vs {m}: {diff:+.2f} EM, p={p:.4f}")
        # learned weights under collusion: agents 0,1 hold one corrupted copy, agent 2 is honest
        w = [r["weights"] for r in runs["epinet2"] if r["condition"] == "collusion" and r.get("weights")]
        if w:
            colluders = sum((x[0][1] + x[1][0]) / 2 for x in w) / len(w)
            honest = sum((x[0][2] + x[1][2]) / 2 for x in w) / len(w)
            lines += ["", f"## Learned peer weights under collusion (n={len(w)})", "",
                      f"- colluder -> colluder w: {colluders:.3f}", f"- colluder -> honest w: {honest:.3f}"]
        for cond in CONDITIONS:
            ws = [r["weights"] for r in runs["epinet2"] if r["condition"] == cond and r.get("weights")]
            if ws:
                m = sum(sum(x[i][j] for i in range(3) for j in range(3) if i != j) / 6 for x in ws) / len(ws)
                lines.append(f"- mean off-diagonal w, {cond}: {m:.3f}")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
