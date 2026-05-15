"""Quick regen of chart 02 with non-overlapping baseline annotation."""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

OUT = Path(__file__).parent / "02_main_performance.png"

# Colors (matching the deck)
COLOR_DARK    = "#0F172A"
COLOR_BLUE    = "#2563EB"
COLOR_GREEN   = "#10B981"
COLOR_RED     = "#EF4444"
COLOR_WEAK    = "#94A3B8"
COLOR_SBERT   = "#9CA3AF"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Six approaches: name, F1, color
data = [
    ("SBERT only",            0.604, COLOR_SBERT),
    ("Weak supervised",       0.440, COLOR_RED),
    ("Self-trained v1",       0.564, COLOR_RED),
    ("Self-trained v2",       0.579, COLOR_RED),
    ("LLM supervised",        0.704, COLOR_BLUE),
    ("Active learning (final)", 0.769, COLOR_GREEN),
]
names  = [d[0] for d in data]
f1s    = [d[1] for d in data]
colors = [d[2] for d in data]

fig, ax = plt.subplots(figsize=(13, 6.5))
bars = ax.bar(names, f1s, color=colors, edgecolor="white", linewidth=2)

# Bar value labels
for bar, v in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=COLOR_DARK)

# Highlight winner
bars[-1].set_edgecolor(COLOR_DARK)
bars[-1].set_linewidth(3)

# SBERT baseline reference line — annotation moved to RIGHT side to avoid bar overlap
ax.axhline(0.604, color=COLOR_WEAK, linestyle=":", alpha=0.45, linewidth=1.5)
ax.text(0.985, 0.612, "SBERT-only baseline (0.604)",
        color=COLOR_WEAK, transform=ax.get_yaxis_transform(),
        fontsize=9, va="bottom", ha="right", style="italic")

ax.set_ylabel("F1 Score on held-out 100-pair gold standard", fontsize=14)
ax.set_title("Final Performance — F1 Across All Six Approaches",
             fontsize=18, pad=15, fontweight="bold")
ax.set_ylim(0, 0.92)
ax.tick_params(axis="x", rotation=8, labelsize=11)
ax.tick_params(axis="y", labelsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Active learning callout
ax.annotate(
    "+0.165 over SBERT\n+0.066 over supervised",
    xy=(5, 0.769), xytext=(4.0, 0.87),
    ha="center", fontsize=11, fontweight="bold", color=COLOR_DARK,
    arrowprops=dict(arrowstyle="->", color=COLOR_DARK, lw=1.5),
)

fig.text(0.5, 0.005,
         "Same 100-pair held-out test set across all approaches. "
         "Labels: 3-LLM consensus (Claude + GPT + Gemini majority vote). "
         "random_state = 42 throughout.",
         ha="center", fontsize=9, style="italic", color="#6b7280")

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"[done] wrote {OUT}")
