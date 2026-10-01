"""Calibration-aware ("RLCD-style") PPO on Breakout, after decider v10's RL stage (decider/docs/RL.md), from the decider repo:

    .venv/bin/python /projects/u6xe/louisk/hardgen/rl_calibrated.py --init MODEL --out RUN [--iters 40 ...]
    .venv/bin/python /projects/u6xe/louisk/hardgen/rl_calibrated.py --init MODEL --out RUN --eval_only   # score + beliefs

Loss per optimizer step:
  actor      PPO clipped surrogate on the sampled action, per-step return baselines (decider.games.rl, unchanged)
  belief     the model is also asked yes/no questions about what happens next ("will the paddle return the ball", "will
             a brick break / a life be lost in the next 30 moves") in the same one-pass format; their outcomes are read
             from the rest of the episode, and the answer is scored with the log score (cross-entropy) against what
             happened - a proper scoring rule, minimised only by stating the true probabilities
  retention  KL(start model || current) on ~1,000 replayed supervised decider questions (the start model's answers are
             computed once, so no second copy of the model is needed in memory)
The actor term and its settings match the plain run (jobs/rl_breakout.sh); belief and retention are added on top.
"""
import argparse
import json
import os
import random
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch
import torch.nn.functional as F

from decider import data as D
from decider.games import envs as G
from decider.games.rl import REWARD_SCALE, NoShuffle, Policy, a_success, returns
from decider.prompt import build

BELIEFS = {"return": "Will the paddle return the ball the next time the ball reaches the bottom?",
           "brick30": "Will at least one brick be broken within the next 30 moves?",
           "life30": "Will a life be lost within the next 30 moves?"}
PADDLE_Y = 160          # a down->up flip of the ball at or below this y is a paddle hit (bricks flip it at y ~72-87)


def rollout(pol, n, rng, max_t, greedy=False, seed0=0, belief_every=4):
    """Breakout episodes with the policy; also records belief questions and resolves their outcomes from the episode."""
    envs = []
    for j in range(n):
        g = G.GAMES["breakout"](); g.reset(seed=seed0 + (j if greedy else rng.randint(0, 10 ** 6)))
        envs.append(dict(g=g, done=False, k=0, opt=g.options[0], items=[], acts=[], rews=[], logps=[], last=g.score(),
                         nopt=len(g.options), pend=[], beliefs=[], prev=g._s(), vy=0))
    pol.m.eval()
    while any(not e["done"] for e in envs):
        due = [e for e in envs if not e["done"]]
        items = []
        for e in due:
            ex, fr = pol.example(e["g"]); items.append(pol.make_item(ex, fr))
            if e["k"] % belief_every == 0:
                s = e["g"]._s(); kinds = ["brick30", "life30"] + (["return"] if s["ball_y"] > 0 else [])
                kind = rng.choice(kinds)
                e["pend"].append(dict(kind=kind, t=e["k"], ctx=f"{e['g'].intro}\n\nSituation: {e['g'].text()}",
                                      score=e["g"].score(), lives=s["lives"]))
        with torch.no_grad():
            lg = pol.logits(items)
        for e, it, row in zip(due, items, lg):
            p = torch.softmax(row[:e["nopt"]].float(), -1).cpu()
            a = int(p.argmax()) if greedy else int(torch.multinomial(p, 1))
            e["items"].append(it); e["acts"].append(a); e["logps"].append(float(torch.log(p[a] + 1e-12)))
            e["rews"].append(0.0); e["opt"] = e["g"].options[a]
        for e in due:
            _, done = e["g"].step(e["opt"]); e["k"] += 1
            sc = e["g"].score(); e["rews"][-1] += (sc - e["last"]) / REWARD_SCALE.get("breakout", 1.0); e["last"] = sc
            if done and e["g"].success():
                e["rews"][-1] += a_success
            s = e["g"]._s(); vy = s["ball_y"] - e["prev"]["ball_y"]
            # a paddle return: the ball turns from down to up near the bottom AND stays in play with no life lost.  When the
            # ball is lost its y resets to 0, which looks like an upward turn - that is a miss, not a return (bug in run 7000530)
            hit = (e["vy"] > 0 and vy < 0 and e["prev"]["ball_y"] >= PADDLE_Y and s["ball_y"] > 0
                   and s["lives"] == e["prev"]["lives"])
            still = []
            for b in e["pend"]:
                out = None
                if b["kind"] == "return":
                    out = True if hit else False if s["lives"] < b["lives"] or done else None
                elif b["kind"] == "brick30":
                    out = True if sc > b["score"] else False if (e["k"] - b["t"] >= 30 or done) else None
                else:
                    out = True if s["lives"] < b["lives"] else False if (e["k"] - b["t"] >= 30 or (done and s["lives"] >= b["lives"])) else None
                if out is None:
                    still.append(b)
                else:
                    e["beliefs"].append(dict(kind=b["kind"], ctx=b["ctx"], gold=int(out)))
            e["pend"] = still; e["prev"] = s; e["vy"] = vy if vy != 0 else e["vy"]
            if done or e["k"] >= max_t:
                e["done"] = True                                     # unresolved beliefs at truncation are dropped
    out = [dict(name="breakout", items=e["items"], acts=e["acts"], rews=e["rews"], logps=e["logps"], score=e["g"].score(),
                nopt=e["nopt"], beliefs=e["beliefs"]) for e in envs]
    for e in envs:
        e["g"].close()
    return out


def belief_items(pol, beliefs):
    # D.Example / D.Q with an explicit task: D.load_cache rebinds __main__.Example / Q (decider.data.core.load_cache),
    # so names imported into this script would silently change type once the replay rows are loaded.
    return [build(D.Example(b["ctx"], [D.Q(BELIEFS[b["kind"]], ["no", "yes"], b["gold"])], "belief"), pol.tok, NoShuffle(),
                  max_ctx_tokens=1024) for b in beliefs]


def calib(p_yes, gold, bins=10):
    """Log score, Brier, top-label ECE (10 bins over confidence 0.5-1), accuracy and base rate of yes/no beliefs."""
    p, y = np.asarray(p_yes, dtype=float), np.asarray(gold)
    ls = float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1))))
    conf = np.maximum(p, 1 - p); correct = ((p >= 0.5) == (y == 1)).astype(float)
    b = np.clip(np.digitize(conf, np.linspace(0.5, 1.0, bins + 1)) - 1, 0, bins - 1)
    ece = sum((b == i).mean() * abs(conf[b == i].mean() - correct[b == i].mean()) for i in range(bins) if (b == i).any())
    return dict(log_score=round(ls, 4), brier=round(float(np.mean((p - y) ** 2)), 4), ece=round(float(ece), 4),
                acc=round(float(correct.mean()), 4), base_rate=round(float(y.mean()), 3), n=int(len(y)))


def belief_probs(pol, items, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            out += torch.softmax(pol.logits(items[i:i + bs]).float()[:, :2], -1)[:, 1].tolist()
    return out


def replay_rows(pol, path, n, rng):
    """n single-question supervised rows from the original mixture, with the start model's log-probs (fixed reference)."""
    train, _ = D.load_cache(path)
    pick = [e for e in rng.sample(train, min(len(train), 20 * n)) if len(e.qs) == 1][:4 * n]
    items = []
    for e in pick:
        it = build(e, pol.tok, random.Random(0), max_options=10, max_ctx_tokens=1024)
        if len(it["ids"]) <= 1024:
            items.append(it)
        if len(items) == n:
            break
    ref = []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            ref += list(F.log_softmax(pol.logits(items[i:i + 32]).float(), -1).cpu())
    del train
    return items, ref


def retention_kl(pol, items, ref):
    lp = F.log_softmax(pol.logits(items).float(), -1)
    r = torch.stack(ref).to(lp.device)
    ok = torch.isfinite(r) & torch.isfinite(lp)
    return (r.exp() * (r - lp)).masked_fill(~ok, 0.0).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--iters", type=int, default=40); ap.add_argument("--envs", type=int, default=16); ap.add_argument("--max_t", type=int, default=300)
    ap.add_argument("--lr", type=float, default=4e-6); ap.add_argument("--gamma", type=float, default=0.97); ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--ppo_steps", type=int, default=4); ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--kl_stop", type=float, default=0.05)
    ap.add_argument("--update_samples", type=int, default=6000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=5); ap.add_argument("--eval_episodes", type=int, default=3)
    ap.add_argument("--belief_w", type=float, default=2.0, help="v10 ratio belief/actor = 0.2/0.1")
    ap.add_argument("--belief_batch", type=int, default=128); ap.add_argument("--belief_every", type=int, default=4)
    ap.add_argument("--ret_w", type=float, default=7.0, help="v10 ratio retention/actor = 0.7/0.1")
    ap.add_argument("--ret_rows", type=int, default=1000); ap.add_argument("--ret_batch", type=int, default=32)
    ap.add_argument("--replay", default="data/mixture_full.pkl")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); logf = open(f"{a.out}/rl.log", "a")
    def log(*s):
        m = " ".join(str(x) for x in s); print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log("[args]", json.dumps(vars(a)))
    rng = random.Random(0); torch.manual_seed(0); G.RENDER = False
    pol = Policy(a.init, False)

    def evaluate(tag):
        trs = rollout(pol, a.eval_episodes, rng, max_t=G.GAMES["breakout"].max_t, greedy=True)
        bel = [b for t in trs for b in t["beliefs"]]
        c = calib(belief_probs(pol, belief_items(pol, bel)), [b["gold"] for b in bel]) if bel else {}
        sc = float(np.mean([t["score"] for t in trs]))
        log(f"[eval] {tag} greedy breakout={sc:.2f} beliefs {json.dumps(c)}"); return sc, c

    if a.eval_only:
        sc, c = evaluate("eval_only")
        trs = rollout(pol, a.envs, rng, a.max_t)                      # sampled play, as in training
        bel = [b for t in trs for b in t["beliefs"]]
        cs = calib(belief_probs(pol, belief_items(pol, bel)), [b["gold"] for b in bel])
        by = {k: calib(belief_probs(pol, belief_items(pol, [b for b in bel if b["kind"] == k])),
                       [b["gold"] for b in bel if b["kind"] == k]) for k in BELIEFS if any(b["kind"] == k for b in bel)}
        log(f"[eval] sampled breakout={np.mean([t['score'] for t in trs]):.2f} beliefs {json.dumps(cs)} by kind {json.dumps(by)}")
        json.dump(dict(greedy=sc, greedy_beliefs=c, sampled_beliefs=cs, by_kind=by), open(f"{a.out}/eval_only.json", "w"), indent=1)
        return

    t0 = time.time(); rep_items, rep_ref = replay_rows(pol, a.replay, a.ret_rows, rng)
    log(f"[retention] {len(rep_items)} replay rows, reference log-probs from the start model in {time.time() - t0:.0f}s")
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, betas=(0.9, 0.95))
    base, _ = evaluate("start"); best = base
    for it in range(1, a.iters + 1):
        t0 = time.time(); trajs = rollout(pol, a.envs, rng, a.max_t, belief_every=a.belief_every); t_roll = time.time() - t0
        items, acts, advs, oldlp = [], [], [], []
        for tr in trajs:
            tr["G"] = returns(tr["rews"], a.gamma)
        T = max(len(tr["G"]) for tr in trajs)
        basel = [float(np.mean([tr["G"][t] for tr in trajs if t < len(tr["G"])])) for t in range(T)]
        sd = float(np.std([tr["G"][t] - basel[t] for tr in trajs for t in range(len(tr["G"]))])) + 1e-6
        for tr in trajs:
            for t in range(len(tr["G"])):
                items.append(tr["items"][t]); acts.append(tr["acts"][t]); advs.append((tr["G"][t] - basel[t]) / sd); oldlp.append(tr["logps"][t])
        bel = [b for tr in trajs for b in tr["beliefs"]]; bitems = belief_items(pol, bel); bgold = [b["gold"] for b in bel]
        pre = calib(belief_probs(pol, bitems), bgold) if bel else {}
        order = list(range(len(items))); rng.shuffle(order); order = order[:a.update_samples]
        pol.m.train(); chunks = [order[i::a.ppo_steps] for i in range(a.ppo_steps)]
        kl_last = pg_tot = bl_tot = rk_tot = 0.0; nb = nsteps = 0; gn = 0.0
        for chunk in chunks:
            kls = []
            for i in range(0, len(chunk), a.batch):
                idx = chunk[i:i + a.batch]
                logp = F.log_softmax(pol.logits([items[j] for j in idx]).float(), -1)
                act = torch.tensor([acts[j] for j in idx], device="cuda"); A = torch.tensor([advs[j] for j in idx], device="cuda")
                lp = logp.gather(1, act[:, None]).squeeze(1); olp = torch.tensor([oldlp[j] for j in idx], device="cuda")
                ratio = torch.exp(lp - olp)
                pg = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                valid = torch.isfinite(logp); lps = logp.masked_fill(~valid, 0.0)
                ent = -(lps.exp().masked_fill(~valid, 0.0) * lps).sum(1).mean()
                ((pg - a.entropy * ent) * len(idx) / len(chunk)).backward()
                pg_tot += pg.item() * len(idx); nb += len(idx); kls.append(float((olp - lp).mean()))
            if bel and a.belief_w > 0:                               # belief: log score against what happened
                bidx = rng.sample(range(len(bitems)), min(a.belief_batch, len(bitems)))
                for i in range(0, len(bidx), a.batch):
                    sub = bidx[i:i + a.batch]
                    lg = pol.logits([bitems[j] for j in sub]).float()[:, :2]
                    ce = F.cross_entropy(lg, torch.tensor([bgold[j] for j in sub], device="cuda"))
                    (a.belief_w * ce * len(sub) / len(bidx)).backward(); bl_tot += ce.item() * len(sub) / len(bidx)
            if a.ret_w > 0:                                          # retention: stay close to the start model
                ridx = rng.sample(range(len(rep_items)), a.ret_batch)
                rk = retention_kl(pol, [rep_items[j] for j in ridx], [rep_ref[j] for j in ridx])
                (a.ret_w * rk).backward(); rk_tot += rk.item()
            nsteps += 1
            gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            kl_last = float(np.mean(kls)) if kls else 0.0
            if kl_last > a.kl_stop:
                break
        sampled = round(float(np.mean([tr["score"] for tr in trajs])), 2)
        log(f"[rl] iter {it}: sampled breakout {sampled} decisions {len(items)} pg {pg_tot / max(1, nb):.3f} kl {kl_last:.4f} "
            f"belief_ce {bl_tot / max(1, nsteps):.3f} retention_kl {rk_tot / max(1, nsteps):.4f} gn {gn:.2f} steps {nsteps} "
            f"rollout {t_roll:.0f}s total {time.time() - t0:.0f}s | beliefs before update {json.dumps(pre)}")
        if it % a.eval_every == 0 or it == a.iters:
            sc, c = evaluate(f"iter {it}")
            if sc > best:
                best = sc; pol.save(f"{a.out}/model"); json.dump(dict(score=sc, beliefs=c, iter=it), open(f"{a.out}/best_eval.json", "w"))
                log(f"[save] best greedy breakout {sc:.2f} -> {a.out}/model")
    pol.save(f"{a.out}/model_last"); log(f"[save] final weights -> {a.out}/model_last")      # the last iterate too


if __name__ == "__main__":
    main()
