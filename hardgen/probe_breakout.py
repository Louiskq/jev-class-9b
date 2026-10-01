"""Print Breakout RAM dynamics under the scripted teacher (CPU only): ball y range, where vertical direction flips, life losses."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from decider.games import envs as G

g = G.GAMES["breakout"](); g.reset(seed=0)
prev = g._s(); prev_vy = 0; flips_up, flips_down, losses, ys = [], [], [], []
for k in range(1500):
    _, done = g.step(g.teacher()); s = g._s(); vy = s["ball_y"] - prev["ball_y"]
    if s["ball_y"] > 0:
        ys.append(s["ball_y"])
    if prev_vy > 0 and vy < 0 and s["ball_y"] > 0:
        flips_up.append(prev["ball_y"])
    if prev_vy < 0 and vy > 0 and s["ball_y"] > 0:
        flips_down.append(prev["ball_y"])
    if s["lives"] < prev["lives"]:
        losses.append((k, prev["ball_y"]))
    prev, prev_vy = s, (vy if vy != 0 else prev_vy)
    if done:
        break
print("steps", k + 1, "score", g.score(), "decide_every", g.decide_every)
print("ball_y min/max", min(ys), max(ys))
print("down->up flips at y:", sorted(flips_up)[:5], "...", sorted(flips_up)[-5:], "n", len(flips_up))
print("up->down flips at y:", sorted(flips_down)[:5], "...", sorted(flips_down)[-5:], "n", len(flips_down))
print("life losses (step, last ball_y):", losses)
