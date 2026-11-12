"""Risk-coverage curves from the per-example dumps of scripts/ood/dump_per_example.py.

Writes:
  results/risk_coverage_curves.csv    selective metrics per threshold (step 0.01)
  results/risk_coverage_summary.csv   per (dataset, method): AURC, metrics at tau in {.7,.8,.9}
  results/figures/risk_coverage_{dataset}.png

    python scripts/ood/risk_coverage.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from uat_lite import metrics
from uat_lite.protocol import METHODS

DATASETS = ["HANS", "ANLI", "SNLI", "IMDb"]
METHOD_KEYS = [k for k, _ in METHODS]
METHOD_LABEL = dict(METHODS)


def threshold_curve(confidences, correct, thresholds):
    """For each threshold, return coverage and accuracy among admitted examples."""
    N = len(confidences)
    rows = []
    for tau in thresholds:
        admitted = confidences >= tau
        n = int(admitted.sum())
        if n == 0:
            cov = 0.0
            acc = float("nan")
            risk = float("nan")
        else:
            cov = n / N
            acc = float(correct[admitted].mean())
            risk = 1.0 - acc
        rows.append({
            "threshold": float(tau),
            "coverage": cov,
            "accuracy": acc,
            "risk": risk,
            "admitted_count": n,
            "total_count": N,
        })
    return rows


def load_per_example(in_dir):
    """Load all .npz files. Returns dict[dataset][method] = dict of arrays + meta."""
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for mk, _ in METHODS:
            p = in_dir / f"{ds}_{mk}.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=False)
            out[ds][mk] = {
                "labels": d["labels"],
                "preds": d["preds"],
                "confidences": d["confidences"],
                "probs": d["probs"],
                "M": int(d["M"]),
                "T": float(d["T"]),
                "N": int(d["N"]),
                "_path": str(p),
            }
    return out





def write_curves_csv(per_ex, thresholds, out_path):
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "method_label", "threshold",
                    "coverage", "accuracy", "risk",
                    "admitted_count", "total_count"])
        for ds, methods in per_ex.items():
            for mk, data in methods.items():
                preds = data["preds"]
                labels = data["labels"]
                conf = data["confidences"]
                correct = (preds == labels).astype(float)
                rows = threshold_curve(conf, correct, thresholds)
                for r in rows:
                    w.writerow([
                        ds, mk, METHOD_LABEL[mk],
                        f"{r['threshold']:.2f}",
                        f"{r['coverage']:.6f}",
                        f"{r['accuracy']:.6f}" if not np.isnan(r["accuracy"]) else "",
                        f"{r['risk']:.6f}" if not np.isnan(r["risk"]) else "",
                        r["admitted_count"],
                        r["total_count"],
                    ])
    print(f"[saved] {out_path}")


def write_summary_csv(per_ex, out_path):
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "method_label",
                    "N", "M", "T",
                    "accuracy", "ece", "nll", "aurc",
                    "best_threshold_region",
                    "coverage_at_0.9", "acc_at_0.9",
                    "coverage_at_0.8", "acc_at_0.8",
                    "coverage_at_0.7", "acc_at_0.7"])
        for ds, methods in per_ex.items():
            for mk, data in methods.items():
                labels = data["labels"]
                preds = data["preds"]
                probs = data["probs"]
                conf = data["confidences"]
                correct = (preds == labels).astype(float)
                accuracy = float(correct.mean())
                ece = metrics.ece(probs, labels)
                nll = metrics.nll(probs, labels)
                aurc = metrics.aurc(conf, correct)

                # threshold metrics + best region
                taus = np.linspace(0.0, 1.0, 101)
                rows = threshold_curve(conf, correct, taus)
                # best region = threshold maximizing accuracy among rows with
                # coverage >= 5% of N (avoid degenerate spikes from tiny admitted sets)
                min_n = max(20, int(0.05 * data["N"]))
                viable = [r for r in rows if r["admitted_count"] >= min_n]
                if viable:
                    best = max(viable, key=lambda r: r["accuracy"])
                    best_region = (f"tau={best['threshold']:.2f} "
                                   f"(cov={best['coverage']:.3f}, "
                                   f"acc={best['accuracy']:.4f})")
                else:
                    best_region = "n/a (no threshold admits >=5% of N)"

                def _at(tau):
                    rr = next(r for r in rows
                              if abs(r["threshold"] - tau) < 1e-9)
                    return rr["coverage"], rr["accuracy"]

                cov9, acc9 = _at(0.90)
                cov8, acc8 = _at(0.80)
                cov7, acc7 = _at(0.70)

                w.writerow([
                    ds, mk, METHOD_LABEL[mk],
                    data["N"], data["M"],
                    f"{data['T']:.4f}" if not np.isnan(data["T"]) else "",
                    f"{accuracy:.6f}",
                    f"{ece:.6f}",
                    f"{nll:.6f}",
                    f"{aurc:.6f}",
                    best_region,
                    f"{cov9:.6f}",
                    f"{acc9:.6f}" if not np.isnan(acc9) else "",
                    f"{cov8:.6f}",
                    f"{acc8:.6f}" if not np.isnan(acc8) else "",
                    f"{cov7:.6f}",
                    f"{acc7:.6f}" if not np.isnan(acc7) else "",
                ])
    print(f"[saved] {out_path}")



def plot_risk_coverage(per_ex, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    method_colors = {
        "base_deterministic": "#888888",
        "base_ts":             "#1f77b4",
        "mc_global":           "#2ca02c",
        "uat_lite":            "#d62728",
        "uat_lite_ts":         "#9467bd",
    }
    method_styles = {
        "base_deterministic": ":",
        "base_ts":             "--",
        "mc_global":           "-.",
        "uat_lite":            "-",
        "uat_lite_ts":         "-",
    }

    saved = []
    for ds, methods in per_ex.items():
        if not methods:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for mk in METHOD_KEYS:
            if mk not in methods:
                continue
            data = methods[mk]
            preds = data["preds"]
            labels = data["labels"]
            conf = data["confidences"]
            correct = (preds == labels).astype(float)
            # full continuous curve sorted by descending confidence
            order = np.argsort(-conf)
            cum = np.cumsum(correct[order])
            N = len(conf)
            coverages = np.arange(1, N + 1) / N
            risks = 1.0 - cum / np.arange(1, N + 1)
            ax.plot(coverages, risks,
                    label=METHOD_LABEL[mk],
                    color=method_colors[mk],
                    linestyle=method_styles[mk],
                    linewidth=1.8)

        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk = 1 - Accuracy")
        ax.set_title(f"Risk-Coverage Curve: {ds}")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, frameon=True)
        plt.tight_layout()
        out_path = out_dir / f"risk_coverage_{ds}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[saved] {out_path}")
        saved.append(out_path)
    return saved



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", default="./results/per_example")
    ap.add_argument("--out_dir", default="./results")
    ap.add_argument("--fig_dir", default="./results/figures")
    args = ap.parse_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    if not in_dir.exists():
        sys.exit(f"[error] input directory not found: {in_dir}")
    per_ex = load_per_example(in_dir)
    print(f"[load] {sum(len(v) for v in per_ex.values())} (dataset, method) entries from {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_curves_csv(per_ex, np.linspace(0.0, 1.0, 101), out_dir / "risk_coverage_curves.csv")
    write_summary_csv(per_ex, out_dir / "risk_coverage_summary.csv")
    for fp in plot_risk_coverage(per_ex, Path(args.fig_dir)):
        print(f"[OK] {fp}")


if __name__ == "__main__":
    main()
