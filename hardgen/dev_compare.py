"""Compare models on the independent dev sets (runs/dev_eval/LABEL/<set>/preds.pkl), with paired tests and seed spread.

    python dev_compare.py --runs /projects/u6xe/louisk/hardgen/runs/dev_eval --ref SFT \
        --group B1=B1,B1_s1,B1_s2 --group B5=B5,B5_s1,B5_s2 --single A,B2,B3,B4,scale2k,...
Per set: accuracy over all questions (pooled over tasks), paired sign test and paired bootstrap 95% CI against --ref.
A --group is one recipe trained with several seeds: reported as mean +- sample sd, and its members are pooled for the
paired test (each member vs ref, then the mean difference).  The seed sd is the noise floor for calling a change real.
"""
import argparse, math, os, pickle
import numpy as np

SETS = ["new_domains", "old_heldout", "public_heldout"]


def correct(runs, label, s):
    p = f"{runs}/{label}/{s}/preds.pkl"
    if not os.path.exists(p):
        return None
    d = pickle.load(open(p, "rb")); out, gold = [], []
    for t in sorted(d):
        P, G = d[t]["probs"], d[t]["golds"]
        out += [int(np.argmax(np.asarray(x)) == g) for x, g in zip(P, G)]; gold += [(t, int(g)) for g in G]
    return np.array(out), gold


def sign_p(fixed, broken):
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b, rng):
    fixed, broken = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    diffs = a - b; boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
    return diffs.mean(), np.percentile(boots, [2.5, 97.5]), fixed, broken, sign_p(fixed, broken)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--ref", default="SFT")
    ap.add_argument("--group", action="append", default=[]); ap.add_argument("--single", default="")
    a = ap.parse_args(); rng = np.random.default_rng(0)
    groups = {g.split("=")[0]: g.split("=")[1].split(",") for g in a.group}
    singles = [x for x in a.single.split(",") if x]
    for s in SETS:
        ref = correct(a.runs, a.ref, s)
        if ref is None:
            print(f"\n## {s}: no reference ({a.ref})"); continue
        print(f"\n## {s}  (n = {len(ref[0])} questions; reference {a.ref} = {ref[0].mean():.3f})\n")
        print("| model | acc | delta vs ref | 95% CI (paired) | fixed / broken | sign p |\n|---|---|---|---|---|---|")
        rows = [(g, m) for g, m in groups.items()] + [(x, [x]) for x in singles]
        for name, members in rows:
            res = [(m, correct(a.runs, m, s)) for m in members]; res = [(m, r) for m, r in res if r is not None]
            if not res:
                continue
            for m, r in res:
                assert r[1] == ref[1], f"{m}: dev items differ from the reference"
            accs = np.array([r[0].mean() for _, r in res])
            if len(res) > 1:   # seeds: mean +- sd, and the member-averaged paired difference
                mean_corr = np.mean([r[0] for _, r in res], axis=0)
                d, ci, f, b, p = paired(mean_corr, ref[0].astype(float), rng)
                f = int(np.mean([((r[0] == 1) & (ref[0] == 0)).sum() for _, r in res])); b = int(np.mean([((r[0] == 0) & (ref[0] == 1)).sum() for _, r in res]))
                p = sign_p(f, b)
                print(f"| {name} (x{len(res)} seeds) | {accs.mean():.3f} +- {accs.std(ddof=1):.3f} | {d:+.3f} | {ci[0]:+.3f} .. {ci[1]:+.3f} | {f} / {b} (mean) | {p:.3f} |")
            else:
                m, r = res[0]; d, ci, f, b, p = paired(r[0].astype(float), ref[0].astype(float), rng)
                print(f"| {name} | {accs[0]:.3f} | {d:+.3f} | {ci[0]:+.3f} .. {ci[1]:+.3f} | {f} / {b} | {p:.3f} |")


if __name__ == "__main__":
    main()
