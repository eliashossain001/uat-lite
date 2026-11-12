"""Calibration and selective-prediction metrics.

All functions take NumPy arrays: ``probs`` is [N, C], ``labels`` is [N].
Confidence is the max softmax probability.
"""

import numpy as np


def _trapz(y, x):
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:  # numpy < 2.0
        return float(np.trapz(y, x))


def ece(probs, labels, num_bins=15):
    """Expected Calibration Error with equal-width bins (lo, hi]."""
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    total, N = 0.0, len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        total += (n_bin / N) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(total)


def brier(probs, labels):
    N = len(labels)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def nll(probs, labels, eps=1e-12):
    true_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_probs, eps, 1.0))))


def accuracy(probs, labels):
    return float((probs.argmax(axis=1) == labels).mean())


def aurc(confidences, correct):
    """Area under the risk-coverage curve, exact (one point per example)."""
    N = len(confidences)
    order = np.argsort(-confidences)
    cum_correct = np.cumsum(correct[order])
    k = np.arange(1, N + 1)
    return _trapz(1.0 - cum_correct / k, k / N)


def aurc_grid(confidences, correct, num_points):
    """AURC evaluated on a fixed coverage grid {1/P, ..., 1}.

    Used by the original training-time evaluation (P=100) and by the global
    MC-dropout baseline script (P=200). Returns (aurc, coverages, risks).
    """
    N = len(confidences)
    sorted_correct = correct[np.argsort(-confidences)]
    coverages = np.arange(1, num_points + 1) / num_points
    risks = []
    for cov in coverages:
        n = max(1, int(N * cov))
        risks.append((n - sorted_correct[:n].sum()) / n)  # error rate
    risks = np.array(risks)
    return _trapz(risks, coverages), coverages, risks


def threshold_metrics(confidences, correct, tau):
    """Selective prediction at a confidence threshold: admit s(x) >= tau."""
    admitted = confidences >= tau
    n_tau = int(admitted.sum())
    acc = float(correct[admitted].sum()) / n_tau if n_tau > 0 else 0.0
    return {"coverage": n_tau / len(confidences), "acc": acc, "n_admitted": n_tau}


def acc_at_coverage(confidences, correct, coverage, rounding="round"):
    """Accuracy on the top-``coverage`` fraction ranked by confidence.

    ``rounding`` controls how the admitted count is computed: ``"round"``
    (OOD scripts) or ``"floor"`` (global MC-dropout script).
    """
    N = len(confidences)
    n = round(N * coverage) if rounding == "round" else int(N * coverage)
    n = max(1, int(n))
    return float(correct[np.argsort(-confidences)[:n]].mean())


def summarize(probs, labels, thresholds=(0.9, 0.8, 0.7), coverages=(0.70, 0.80, 0.90),
              num_bins=15):
    """Standard metric block used by the OOD / selective-prediction tables.

    Keys: accuracy, ece, brier, nll, aurc, cov@τ, acc@τ, n@τ, acc@covXX%.
    """
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    out = {
        "accuracy": float(correct.mean()),
        "ece": ece(probs, labels, num_bins),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
        "aurc": aurc(confidences, correct),
    }
    for tau in thresholds:
        tm = threshold_metrics(confidences, correct, tau)
        key = f"{tau:.1f}"
        out[f"cov@{key}"] = tm["coverage"]
        out[f"acc@{key}"] = tm["acc"]
        out[f"n@{key}"] = tm["n_admitted"]
    for cov in coverages:
        out[f"acc@cov{cov * 100:.0f}%"] = acc_at_coverage(confidences, correct, cov)
    return out


def from_logits(logits, labels, num_bins=15):
    """accuracy / ece / brier / nll from torch logits and labels."""
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=-1).numpy()
    labels = labels.numpy()
    return {"accuracy": accuracy(probs, labels), "ece": ece(probs, labels, num_bins),
            "brier": brier(probs, labels), "nll": nll(probs, labels)}


def training_summary(probs, preds, labels, confidences, num_bins=15):
    """Metric block of the original training-time evaluation (scripts/evaluate/evaluate.py,
    baselines, ensembles, post-hoc calibration).

    ECE bins max-prob confidence against ``preds``; AURC uses ``confidences`` on a
    100-point grid; NLL is clipped at 1e-10; selective accuracy is on the top
    70/80/90/100% most-confident examples.
    """
    from sklearn.metrics import f1_score
    correct = (preds == labels).astype(float)
    max_prob = probs.max(axis=1)
    edges = np.linspace(0, 1, num_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (max_prob > lo) & (max_prob <= hi)
        if m.sum() > 0:
            ece_val += (m.sum() / len(max_prob)) * abs(correct[m].mean() - max_prob[m].mean())
    N, order = len(labels), np.argsort(-confidences)
    sel = {}
    for cov in (0.7, 0.8, 0.9, 1.0):
        n = int(N * cov)
        sel[cov] = float(correct[order[:n]].mean()) if n > 0 else 0.0
    return {
        "ece": float(ece_val),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels, eps=1e-10),
        "aurc": aurc_grid(confidences, correct, 100)[0],
        "accuracy": float(correct.mean()),
        "f1": float(f1_score(labels, preds, average="binary" if len(np.unique(labels)) == 2 else "macro")),
        "selective_acc_70": sel[0.7], "selective_acc_80": sel[0.8],
        "selective_acc_90": sel[0.9], "selective_acc_100": sel[1.0],
    }
