"""CPU check of rl_calibrated's 'return' labelling: play with a noisy teacher (so balls are missed too) and compare the old
and new hit rules against life losses."""
import os, random
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from decider.games import envs as G
rng = random.Random(0); stats = {"old": [0, 0], "new": [0, 0]}
for ep in range(4):
    g = G.GAMES["breakout"](); g.reset(seed=ep); prev = g._s(); pvy = 0; pend = {"old": [], "new": []}
    for k in range(1500):
        opt = g.teacher() if rng.random() < 0.7 else rng.choice(g.options)
        if k % 4 == 0 and prev["ball_y"] > 0:
            for r in pend: pend[r].append(prev["lives"])
        _, done = g.step(opt); s = g._s(); vy = s["ball_y"] - prev["ball_y"]
        old = pvy > 0 and vy < 0 and prev["ball_y"] >= 160
        new = old and s["ball_y"] > 0 and s["lives"] == prev["lives"]
        for r, h in (("old", old), ("new", new)):
            still = []
            for lv in pend[r]:
                if h: stats[r][0] += 1
                elif s["lives"] < lv or done: stats[r][1] += 1
                else: still.append(lv)
            pend[r] = still
        prev, pvy = s, (vy if vy != 0 else pvy)
        if done: break
    g.close()
for r, (y, n) in stats.items():
    print(f"{r} rule: returned {y}, missed {n}, yes-rate {y / max(1, y + n):.2f}")
