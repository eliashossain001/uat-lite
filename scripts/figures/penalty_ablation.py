"""
ACL-style grouped bar chart for penalty functional form ablation.
Saves to PDF for direct inclusion in LaTeX paper.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────
methods  = ["Exponential", "Linear", "Reciprocal"]
metrics  = ["ECE ↓", "Acc ↑", "NLL ↓", "AURC ↓"]

# rows = methods, cols = metrics
data = np.array([
    [0.3414, 0.504, 1.4343, 0.5098],  # Exponential
    [0.3260, 0.502, 1.3942, 0.5126],  # Linear
    [0.3470, 0.505, 1.4337, 0.5095],  # Reciprocal
])

# Bold (best) values from the table
# ECE↓  → Linear       (col 0)
# Acc↑  → Reciprocal   (col 1)
# NLL↓  → Linear       (col 2)
# AURC↓ → Reciprocal   (col 3)
best_mask = np.array([
    [False, False, False, False],
    [True,  False, True,  False],
    [False, True,  False, True ],
])

# ── Style ────────────────────────────────────────────────────────────────────
# ACL single-column width ≈ 3.33 in; two-column ≈ 7.0 in
FIG_W, FIG_H = 6.5, 3.4   # fits in ACL two-column or as wide figure

# Muted, publication-quality palette (distinct from the teal/gray of the ICML fig)
PALETTE = {
    "Exponential": "#4878CF",   # muted blue
    "Linear":      "#D65F5F",   # muted red / brick
    "Reciprocal":  "#6AAB6A",   # muted green
}
HATCH = "///"     # best-bar hatch pattern
EDGE  = "black"
BAR_W = 0.22
GAP   = 0.06      # extra gap between metric clusters

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "legend.fontsize":  8,
    "xtick.labelsize":  9,
    "ytick.labelsize":  8,
    "pdf.fonttype":     42,   # embed fonts for camera-ready PDFs
    "ps.fonttype":      42,
})

n_methods  = len(methods)
n_metrics  = len(metrics)
group_w    = n_methods * BAR_W + GAP
x_centers  = np.arange(n_metrics) * group_w

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# ── Draw bars ────────────────────────────────────────────────────────────────
offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * BAR_W

for m_idx, method in enumerate(methods):
    xs     = x_centers + offsets[m_idx]
    vals   = data[m_idx]
    color  = PALETTE[method]

    for g_idx, (x, v, is_best) in enumerate(zip(xs, vals, best_mask[m_idx])):
        hatch = HATCH if is_best else None
        ax.bar(
            x, v,
            width=BAR_W,
            color=color,
            edgecolor=EDGE,
            linewidth=0.6,
            hatch=hatch,
            zorder=3,
        )

# ── Axes formatting ──────────────────────────────────────────────────────────
ax.set_xticks(x_centers)
ax.set_xticklabels(metrics)
ax.set_xlabel("Metric", labelpad=6)
ax.set_ylabel("Score", labelpad=6)
ax.set_title(
    "Penalty Functional Form Comparison on SQuAD 2.0\n"
)

ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
ax.set_axisbelow(True)

# ── Legend ───────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=PALETTE[m], edgecolor=EDGE, linewidth=0.6, label=m)
    for m in methods
]
# add a "best" indicator patch
best_patch = mpatches.Patch(
    facecolor="white", edgecolor=EDGE, linewidth=0.6,
    hatch=HATCH, label="Best per metric"
)
handles.append(best_patch)

ax.legend(
    handles=handles,
    loc="upper left",
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
    ncol=2,
)

# ── Subtle value labels on top of each bar ───────────────────────────────────
for m_idx, method in enumerate(methods):
    xs   = x_centers + offsets[m_idx]
    vals = data[m_idx]
    for x, v in zip(xs, vals):
        ax.text(
            x, v + 0.008, f"{v:.4f}",
            ha="center", va="bottom",
            fontsize=5.5, color="#333333", rotation=90,
        )

fig.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
out_pdf = "results/figures/penalty_ablation.pdf"
out_png = "results/figures/penalty_ablation.png"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
fig.savefig(out_png, format="png", bbox_inches="tight", dpi=300)
print(f"Saved: {out_pdf}  {out_png}")