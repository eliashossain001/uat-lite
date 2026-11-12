"""Five-method evaluation protocol shared by the OOD and shift experiments.

Methods (keys used in all result JSONs):
  base_deterministic  single pass, dropout off
  base_ts             base + temperature fitted on the in-distribution dev set
  mc_global           global MC dropout, mean of softmax (inference.predict_mc_dropout)
  uat_lite            UAT-LITE, estimator chosen by ``uat`` (inference.UAT_ESTIMATORS)
  uat_lite_ts         uat_lite + temperature fitted on dev MC-mean logits
"""

import torch
import torch.nn.functional as F

from .calibration import fit_temperature
from .inference import DEFAULT_UAT, UAT_ESTIMATORS, predict_deterministic, predict_mc_dropout
from .metrics import summarize

METHODS = [
    ("base_deterministic", "Baseline"),
    ("base_ts", "Base+TS"),
    ("mc_global", "MC Dropout"),
    ("uat_lite", "UAT-LITE"),
    ("uat_lite_ts", "UAT-LITE+TS"),
]

METHOD_NOTES = {
    "base_deterministic": "model.eval(), single forward, no dropout",
    "base_ts": "T fitted on in-distribution dev logits",
    "mc_global": "ALL dropout active, M passes, mean-of-softmax",
    "uat_lite_ts": "UAT-LITE + T fitted on dev MC-mean logits",
}
UAT_NOTES = {
    "attention": "forward_uat: uncertainty-weighted attention (Q-only), lagged online U, dropout 0.1/0.2/0.3",
    "embedding": "forward_with_mc_sampling: embedding dropout only, no attention modulation",
    "mc_logits": "all dropout active, softmax of mean logits",
}


def method_notes(uat):
    return {**METHOD_NOTES, "uat_lite": UAT_NOTES[uat]}


def nli3_to_2(probs):
    """[N,3] MNLI probs -> [N,2] HANS probs: (entailment, neutral + contradiction)."""
    return torch.cat([probs[:, 0:1], probs[:, 1:2] + probs[:, 2:3]], dim=1)


def run_five_methods(model, test_loader, dev_loader, M, device, map_to_2class=False,
                     evaluate=summarize, return_probs=False, uat=DEFAULT_UAT):
    """Run all five methods. Temperatures are fitted on ``dev_loader``.

    ``map_to_2class`` maps 3-class NLI predictions to HANS's 2 classes *after*
    softmax / temperature scaling (temperatures are fitted in 3-class space).
    Returns (results, test_labels) or, with ``return_probs``, also a dict of the
    per-method probability arrays.
    """
    to_np = (lambda p: nli3_to_2(p).numpy()) if map_to_2class else (lambda p: p.numpy())
    probs, temps = {}, {}

    print("  [1/5] Base deterministic")
    logits, labels = predict_deterministic(model, test_loader, device)
    labels = labels.numpy()
    probs["base_deterministic"] = to_np(F.softmax(logits, dim=-1))

    print("  [2/5] Base + TS")
    dev_logits, dev_labels = predict_deterministic(model, dev_loader, device)
    temps["base_ts"] = fit_temperature(dev_logits, dev_labels)
    probs["base_ts"] = to_np(F.softmax(logits / temps["base_ts"], dim=-1))

    print(f"  [3/5] MC dropout (M={M})")
    mc_probs, _ = predict_mc_dropout(model, test_loader, M, device)
    probs["mc_global"] = to_np(mc_probs)

    print(f"  [4/5] UAT-LITE ({uat}, M={M})")
    uat_fn = UAT_ESTIMATORS[uat]
    uat_logits, uat_probs, _ = uat_fn(model, test_loader, M, device)
    probs["uat_lite"] = to_np(uat_probs)

    print("  [5/5] UAT-LITE + TS")
    dev_uat_logits, _, dev_uat_labels = uat_fn(model, dev_loader, M, device)
    temps["uat_lite_ts"] = fit_temperature(dev_uat_logits, dev_uat_labels)
    probs["uat_lite_ts"] = to_np(F.softmax(uat_logits / temps["uat_lite_ts"], dim=-1))

    results = {}
    for key, _ in METHODS:
        results[key] = evaluate(probs[key], labels)
        if key in temps:
            results[key]["T"] = temps[key]
    if return_probs:
        return results, labels, probs, temps
    return results, labels


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_threshold_table(title, results, n):
    print(f"\n{'=' * 120}\n  {title}  (N={n})\n{'=' * 120}")
    print("  {:>14s}  {:>6s}  {:>6s}  {:>6s}".format("Method", "Acc", "ECE", "NLL")
          + "".join(f"  {'Cov@.' + t:>7s}  {'Acc@.' + t:>7s}  {'N@.' + t:>5s}" for t in "987")
          + f"  {'AURC':>7s}")
    for key, label in METHODS:
        m = results[key]
        row = f"  {label:>14s}  {m['accuracy']:>6.4f}  {m['ece']:>6.4f}  {m['nll']:>6.4f}"
        for t in ("0.9", "0.8", "0.7"):
            row += f"  {m['cov@' + t]:>7.4f}  {m['acc@' + t]:>7.4f}  {m['n@' + t]:>5d}"
        print(row + f"  {m['aurc']:>7.4f}")


def print_fixed_coverage_table(title, results, n):
    print(f"\n{'=' * 85}\n  {title} -- fixed-coverage selective prediction  (N={n})\n{'=' * 85}")
    print("  {:>14s}  {:>6s}  {:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>7s}".format(
        "Method", "Acc", "ECE", "Acc@90%", "Acc@80%", "Acc@70%", "AURC"))
    for key, label in METHODS:
        m = results[key]
        print(f"  {label:>14s}  {m['accuracy']:>6.4f}  {m['ece']:>6.4f}  {m['acc@cov90%']:>10.4f}"
              f"  {m['acc@cov80%']:>10.4f}  {m['acc@cov70%']:>10.4f}  {m['aurc']:>7.4f}")


def print_latex_rows(name, results):
    """LaTeX rows: threshold table, then fixed-coverage table."""
    print(f"\n% LaTeX: {name} threshold-based")
    for key, label in METHODS:
        m = results[key]
        cells = [m["accuracy"], m["ece"], m["nll"]]
        row = f"        {label} & " + " & ".join(f"{v:.4f}" for v in cells)
        for t in ("0.9", "0.8", "0.7"):
            row += f" & {m['cov@' + t]:.4f} & {m['acc@' + t]:.4f} & {m['n@' + t]:d}"
        print(row + f" & {m['aurc']:.4f} \\\\")
    print(f"\n% LaTeX: {name} fixed-coverage")
    for key, label in METHODS:
        m = results[key]
        cells = [m["accuracy"], m["ece"], m["acc@cov90%"], m["acc@cov80%"], m["acc@cov70%"], m["aurc"]]
        print(f"        {label} & " + " & ".join(f"{v:.4f}" for v in cells) + " \\\\")
