"""Render the README figures from results/results.json."""
import json, math, pathlib
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())
OUT = HERE.parent / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a7a1"
plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)

# 1. JevBench hard accuracy with binomial standard error
H = R["jevbench_hard"]; n = H["n_items"]
names = sorted(H["acc"], key=H["acc"].get)
fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE); style(ax)
for i, k in enumerate(names):
    p = H["acc"][k]; se = math.sqrt(p * (1 - p) / n)
    c = GRAY if k == "SFT" else ORANGE if p < H["acc"]["SFT"] - 0.05 else BLUE
    ax.barh(i, p, height=0.6, color=c, zorder=2)
    ax.errorbar(p, i, xerr=se, color=INK, lw=1.2, capsize=3, zorder=3)
    ax.text(p + se + 0.008, i, f"{p:.3f}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.axvline(H["acc"]["SFT"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.set_xlim(0.5, 0.82)
ax.set_xlabel(f"JevBench hard accuracy (n={n}, bars show ±1 s.e.)")
ax.set_title("Most text-LoRA variants sit within noise of the SFT baseline; code-only data hurts",
             loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(OUT / "jevbench_hard.png", dpi=160); plt.close(fig)

# 2. Breakout: greedy score, sampled score, belief calibration (ECE)
B = R["breakout"]; models = [k for k in B if k != "source"]
panels = [("greedy", "Greedy score (10 games)"), ("sampled", "Sampled score (16 games)"),
          ("belief_ece", "Belief ECE (lower is better)")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor=SURFACE)
for ax, (key, title) in zip(axes, panels):
    style(ax)
    for i, m in enumerate(models):
        v = B[m][key]
        ax.barh(i, v, height=0.6, color=GRAY if m == "SFT" else BLUE, zorder=2)
        ax.text(v, i, f" {v:g}", va="center", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models if ax is axes[0] else [])
    ax.invert_yaxis(); ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlim(0, max(B[m][key] for m in models) * 1.25)
fig.tight_layout(); fig.savefig(OUT / "breakout.png", dpi=160); plt.close(fig)
