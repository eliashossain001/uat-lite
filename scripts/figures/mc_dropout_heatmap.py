"""Global MC-dropout diagnostics heatmap. Values are hard-coded: rows Acc..Acc@90% come from
results/{squad,mnli,sst2}_mc_global_M10.json; the last two rows compare against Base / Base+TS."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# 1) Data
# -----------------------------
rows = [
    ("Acc", "Higher", [0.6555, 0.7101, 0.9014, 0.7557]),
    ("ECE", "Lower",  [0.1600, 0.0565, 0.0501, 0.0889]),
    ("Brier", "Lower",[0.5119, 0.3969, 0.1485, 0.3524]),
    ("NLL", "Lower",  [0.9165, 0.6791, 0.2800, 0.6252]),
    ("AURC","Lower",  [0.3125, 0.1475, 0.0258, 0.1619]),
    ("Acc@70%", "Higher",[0.6951,0.8000,0.9639,0.8197]),
    ("Acc@80%", "Higher",[0.6833,0.7697,0.9569,0.8033]),
    ("Acc@90%", "Higher",[0.6712,0.7410,0.9362,0.7828]),
    ("ΔECE vs. Base", "Lower",[-0.0268,-0.0251,-0.0030,-0.0183]),
    ("Gap to TS (ECE)","Lower",[0.0812,0.0417,0.0338,0.0522]),
]
cols = ["SQuAD 2.0", "MNLI", "SST-2", "Avg"]

df = pd.DataFrame({
    "Metric":[r[0] for r in rows],
    "Goal":[r[1] for r in rows],
    **{cols[i]:[r[2][i] for r in rows] for i in range(4)}
})
val = df[cols].to_numpy(float)

# -----------------------------
# 2) Row-normalize to [0,1] (goal-aware)
# -----------------------------
score = np.zeros_like(val)
eps = 1e-12
for i in range(val.shape[0]):
    v = val[i]
    vmin, vmax = v.min(), v.max()
    if df["Goal"].iloc[i] == "Higher":
        score[i] = (v - vmin) / (vmax - vmin + eps)
    else:
        score[i] = (vmax - v) / (vmax - vmin + eps)
score = score ** 0.95  # gentle contrast

# -----------------------------
# 3) Soft diverging palette (no neon)
# -----------------------------
nyt_soft = LinearSegmentedColormap.from_list(
    "nyt_soft",
    [(0.00, "#5B9BEE"), (0.50, "#B79AB8"), (1.00, "#F06A6A")]
)

# -----------------------------
# 4) Plot (fixed colorbar label)
# -----------------------------
fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
bg = "#F3F6FB"
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

im = ax.imshow(score, cmap=nyt_soft, vmin=0, vmax=1, aspect="auto")

# ticks/labels
ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols, fontsize=12)
ax.set_yticks(np.arange(len(df)))
ax.set_yticklabels(df["Metric"].tolist(), fontsize=12)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

# white gridlines
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(df), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.5)
ax.tick_params(which="minor", bottom=False, left=False)

# remove spines
for s in ax.spines.values():
    s.set_visible(False)

# annotations (raw values)
def fmt(metric, x):
    if "Δ" in metric or "Gap" in metric:
        return f"{x:+.4f}"
    return f"{x:.4f}"

for i in range(val.shape[0]):
    for j in range(val.shape[1]):
        s = score[i, j]
        txt_color = "white" if s > 0.62 else "#1A1A1A"
        ax.text(j, i, fmt(df["Metric"].iloc[i], val[i, j]),
                ha="center", va="center", fontsize=11,
                color=txt_color, fontweight=600)

ax.set_title("Global MC Dropout Diagnostics (M=10) — BERT-base",
             fontsize=16, pad=18, fontweight=600)

# ---- Colorbar: label ABOVE the bar (prevents overflow)
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cbar.set_ticks([0, 0.5, 1.0])
cbar.ax.set_yticklabels(["0", "0.5", "1"])
cbar.outline.set_linewidth(0.6)
cbar.ax.tick_params(length=0)

# Put label above, centered (NYT-like)
cbar.ax.set_title("Row-normalized\nbetter →", fontsize=12, pad=10)

# -----------------------------
# 5) Save
# -----------------------------
plt.savefig("results/figures/mc_dropout_heatmap.pdf", bbox_inches="tight")
plt.savefig("results/figures/mc_dropout_heatmap.png", dpi=300, bbox_inches="tight")
print("Saved: mc_dropout_heatmap_nyt_soft_fixed.pdf")