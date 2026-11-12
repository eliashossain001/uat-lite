"""MNLI matched -> mismatched shift (Table 3): ID ECE, OOD ECE, delta ECE, average ECE,
mean ± std over seeds. Each seed re-draws the dev/test split of validation_matched;
the dev half is used for temperature fitting, the full mismatched set is OOD.

    python scripts/evaluate/mnli_shift.py --seeds 42 43 44 45 46
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from uat_lite import data, metrics
from uat_lite.calibration import fit_temperature
from uat_lite.inference import UAT_ESTIMATORS, load_checkpoint, predict_deterministic, predict_mc_dropout
from uat_lite.utils import add_common_args, ckpt, save_json

METHODS = ["Base", "Base+TS", "MC Dropout", "UAT-LITE", "UAT-LITE+TS"]
KEYS = ["id_ece", "ood_ece", "delta_ece", "avg_ece", "id_acc", "ood_acc"]


def build_loaders(tok, max_length, batch_size, seed):
    dev, id_test = data.dev_test_loaders("mnli", tok, max_length, batch_size, 0.5, seed)
    mm = load_dataset("glue", "mnli")["validation_mismatched"]
    ood = data.make_loader(data.encode(tok, list(mm["premise"]), list(mm["hypothesis"]), max_length),
                           np.array(mm["label"]), batch_size)
    return dev, id_test, ood


def method_probs(method, model, dev, id_test, ood, M, device, uat):
    """(probs_id, labels_id, probs_ood, labels_ood, T or None) for one method."""
    T = None
    if method in ("Base", "Base+TS"):
        if method == "Base+TS":
            T = fit_temperature(*predict_deterministic(model, dev, device))
        run = lambda loader: predict_deterministic(model, loader, device)
        post = lambda logits: F.softmax(logits if T is None else logits / T, dim=-1)
    elif method == "MC Dropout":
        run = lambda loader: predict_mc_dropout(model, loader, M, device)
        post = lambda probs: probs
    elif method == "UAT-LITE":
        run = lambda loader: UAT_ESTIMATORS[uat](model, loader, M, device)[1:]
        post = lambda probs: probs
    else:  # UAT-LITE+TS
        dev_logits, _, dev_labels = UAT_ESTIMATORS[uat](model, dev, M, device)
        T = fit_temperature(dev_logits, dev_labels)
        run = lambda loader: UAT_ESTIMATORS[uat](model, loader, M, device)[::2]
        post = lambda logits: F.softmax(logits / T, dim=-1)
    x_id, y_id = run(id_test)
    x_ood, y_ood = run(ood)
    return post(x_id).numpy(), y_id.numpy(), post(x_ood).numpy(), y_ood.numpy(), T


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__), ckpt("mnli"),
                        "./results/mnli_shift_table3.json")
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = p.parse_args()

    model = load_checkpoint(args.ckpt_path, args.device)
    tok = AutoTokenizer.from_pretrained(model.base_model_name)
    per_seed = {m: [] for m in METHODS}
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        loaders = build_loaders(tok, args.max_length, args.batch_size, seed)
        for method in METHODS:
            t0 = time.time()
            p_id, y_id, p_ood, y_ood, T = method_probs(method, model, *loaders, args.M, args.device, args.uat)
            r = {"method": method}
            if T is not None:
                r["T"] = T
            r["id_ece"], r["ood_ece"] = metrics.ece(p_id, y_id, args.bins), metrics.ece(p_ood, y_ood, args.bins)
            r["delta_ece"], r["avg_ece"] = r["ood_ece"] - r["id_ece"], 0.5 * (r["id_ece"] + r["ood_ece"])
            r["id_acc"], r["ood_acc"] = metrics.accuracy(p_id, y_id), metrics.accuracy(p_ood, y_ood)
            r["seed"], r["elapsed_s"] = seed, round(time.time() - t0, 1)
            per_seed[method].append(r)
            print(f"  {method:>12s}  ID ECE={r['id_ece']:.4f}  OOD ECE={r['ood_ece']:.4f}  "
                  f"dECE={r['delta_ece']:+.4f}  Avg={r['avg_ece']:.4f}")

    summary, rows = {}, []
    print(f"\n  mean ± std over {len(args.seeds)} seeds")
    for method in METHODS:
        agg = {k: {"mean": float(np.mean([r[k] for r in per_seed[method]])),
                   "std": float(np.std([r[k] for r in per_seed[method]]))} for k in KEYS}
        summary[method] = {"per_seed": per_seed[method], "aggregate": agg}
        print(f"  {method:>12s}  " + "  ".join(f"{k}={agg[k]['mean']:.4f}±{agg[k]['std']:.4f}" for k in KEYS))
        cell = lambda k, sign="": f"${sign}{agg[k]['mean']:.4f}{{\\scriptstyle\\pm {agg[k]['std']:.4f}}}$"
        rows.append(f"        {method} & {cell('id_ece')} & {cell('ood_ece')} & "
                    f"{cell('delta_ece', '+' if agg['delta_ece']['mean'] >= 0 else '')} & {cell('avg_ece')} \\\\")
    print("\n        Method & ID ECE & OOD ECE & $\\Delta$ECE & Avg ECE \\\\\n" + "\n".join(rows))

    save_json({"config": {"uat": args.uat, "M": args.M, "bins": args.bins, "seeds": args.seeds, "ckpt_path": args.ckpt_path,
                          "max_length": args.max_length, "batch_size": args.batch_size, "dev_fraction": 0.5},
               "methods": summary}, args.output_json)
    csv = Path(args.output_json).with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("method,seed," + ",".join(KEYS) + "\n")
        for method in METHODS:
            for r in per_seed[method]:
                f.write(f"{method},{r['seed']}," + ",".join(f"{r[k]:.6f}" for k in KEYS) + "\n")
    print(f"[OK] saved {csv}")


if __name__ == "__main__":
    main()
