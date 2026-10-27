"""Pooled paired tests: per-episode EM averaged over seeds, bootstrap over questions."""
import collections, glob, json, os, random, re, sys
root = sys.argv[1]
R = collections.defaultdict(dict)
for p in glob.glob(os.path.join(root, "*", "test_rows.jsonl")):
    n = os.path.basename(os.path.dirname(p)); m = re.match(r"(.+)_s(\d+)$", n)
    R[m.group(1)][int(m.group(2))] = {(r["dataset"], r["qid"], r["regime"]): r for r in map(json.loads, open(p, encoding="utf8"))}
def avg(v, seeds):
    out = {}
    for k in next(iter(R[v].values())):
        out[k] = sum(R[v][s][k]["em"] for s in seeds) / len(seeds)
    return out
def test(a, b, seeds, pred, draws=5000):
    A, B = avg(a, seeds), avg(b, seeds)
    by = collections.defaultdict(list)
    for k in A:
        if pred(k): by[(k[0], k[1])].append(A[k] - B[k])
    qs = list(by); tot = lambda s: sum(sum(by[q]) for q in s) / sum(len(by[q]) for q in s)
    rng = random.Random(0); bs = sorted(tot([qs[rng.randrange(len(qs))] for _ in qs]) for _ in range(draws))
    return 100 * tot(qs), 100 * bs[int(.025 * draws)], 100 * bs[int(.975 * draws)]
out = {}
S = lambda v, w: sorted(set(R[v]) & set(R[w]))
for a, b in [("quote","ic"),("sg","ic"),("belief","ic"),("slots","ic"),("free","ic"),("quote","belief"),("quote","slots"),("quote","free"),("quote","sg")]:
    seeds = S(a, b)
    for reg in ("split", "colocated"):
        out[f"{a}-{b}/{reg}"] = test(a, b, seeds, lambda k, reg=reg: k[2] == reg) + (len(seeds),)
        print(f"{a:7s}-{b:7s} {reg:9s} d={out[f'{a}-{b}/{reg}'][0]:+5.2f} [{out[f'{a}-{b}/{reg}'][1]:+.2f},{out[f'{a}-{b}/{reg}'][2]:+.2f}] seeds={len(seeds)}")
json.dump(out, open(os.path.join(root, "..", "pooled.json"), "w"), indent=1)
