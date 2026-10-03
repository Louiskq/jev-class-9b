"""Side-by-side JevBench + regression comparison of decider9b runs (numbers only)."""
import collections
import json
import os

JB = "/projects/u6xe/louisk/jevbench-runs"
RUNS = [("SFT", "RUN-decider9b-6951699", "/projects/u6xe/louisk/decider/runs/decider9b_full"),
        ("A", "RUN-decider9b-loraA-6961557", "/projects/u6xe/louisk/hardgen/runs/loraA"),
        ("B1", "RUN-decider9b-loraB1-6970139", "/projects/u6xe/louisk/hardgen/runs/loraB1"),
        ("B2", "RUN-decider9b-loraB2-6970565", "/projects/u6xe/louisk/hardgen/runs/loraB2"),
        ("B3", "RUN-decider9b-loraB3-7005857", "/projects/u6xe/louisk/hardgen/runs/loraB3"),
        ("B4", "RUN-decider9b-loraB4-7038103", "/projects/u6xe/louisk/hardgen/runs/loraB4"),
        ("B5", "RUN-decider9b-loraB5-7037893", "/projects/u6xe/louisk/hardgen/runs/loraB5"),
        ("RLplain", "RUN-decider9b-rl-plain-7036515", "/projects/u6xe/louisk/hardgen/runs/rl_breakout"),
        ("RLCDv2", "RUN-decider9b-rlcd-v2-7036514", "/projects/u6xe/louisk/hardgen/runs/rl_breakout_cal_v2"),
        ("combo", "RUN-decider9b-combo-rlcd-b1-7036551", "/projects/u6xe/louisk/hardgen/runs/combo_rlcd_b1")]


def ece(d):
    e = d["ece"]
    return e.get("ece") if isinstance(e, dict) else e


print("%-4s %6s %5s | %-15s %-15s %-15s | %-23s %-23s %5s" % ("run", "items", "ok", "easy acc/ece", "orig acc/ece",
                                                            "HARD acc/ece", "regr IN acc/nll/ece", "regr OUT acc/nll/ece", "T"))
fam = collections.defaultdict(dict)
for name, jb, run in RUNS:
    rs = [json.loads(l) for l in open(f"{JB}/{jb}/results.jsonl") if l.strip()]
    s = {t: json.load(open(f"{JB}/{jb}/summary-{t}.json")) for t in ("easy", "original", "hard")}
    reg = json.load(open(f"{run}/eval/regression_state_first/eval.json"))["agg"] if os.path.exists(f"{run}/eval/regression_state_first/eval.json") else None
    cfg = json.load(open(f"{run}/model/decider_config.json"))
    f = lambda d: "%.3f/%.3f/%.3f" % (d["acc"], d["nll"], d["ece"])
    print("%-4s %6d %5d | %.3f / %.3f   %.3f / %.3f   %.3f / %.3f   | %-23s %-23s %5.2f" % (
        name, len(rs), sum(bool(r["ok"]) for r in rs), s["easy"]["accuracy"], ece(s["easy"]), s["original"]["accuracy"],
        ece(s["original"]), s["hard"]["accuracy"], ece(s["hard"]), f(reg["in_task"]) if reg else "-", f(reg["heldout"]) if reg else "-",
        cfg["temperature"]))
    for l in open(f"{JB}/{jb}/results-hard.jsonl"):
        r = json.loads(l); c = fam[r["family"]].setdefault(name, [0, 0]); c[0] += bool(r["correct"]); c[1] += 1
print()
print("%-18s" % "hard family" + "".join("%9s" % n for n, _, _ in RUNS))
for k in sorted(fam, key=lambda k: fam[k]["SFT"][0] / fam[k]["SFT"][1]):
    print("%-18s" % k + "".join("%9s" % ("%d/%d" % tuple(fam[k][n])) for n, _, _ in RUNS))


if __name__ == "__main__":
    import math
    def res(jb):
        return {json.loads(l)["task_id"]: bool(json.loads(l)["correct"]) for l in open(f"{JB}/{jb}/results-hard.jsonl")}
    R = {n: res(jb) for n, jb, _ in RUNS}
    print()
    for ref in ("SFT", "B1"):
        for n, _, _ in RUNS:
            if n in (ref, "SFT") or (ref == "B1" and n in ("A",)):
                continue
            g = sum(1 for k in R[ref] if R[n][k] and not R[ref][k]); l = sum(1 for k in R[ref] if R[ref][k] and not R[n][k])
            m, t = min(g, l), g + l
            p = min(1.0, 2 * sum(math.comb(t, i) for i in range(m + 1)) / 2 ** t) if t else 1.0
            print("%-4s -> %-7s: %2d fixed %2d broken, net %+3d, sign p = %.3f" % (ref, n, g, l, g - l, p))
