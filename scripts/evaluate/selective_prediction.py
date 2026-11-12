"""Threshold-based selective prediction (Coverage/Acc/N at tau in {0.9,0.8,0.7}, AURC)
on the in-distribution test half, five methods.

The submitted selective-prediction table used ``--uat mc_logits`` (global MC
dropout with logit averaging).

    python scripts/evaluate/selective_prediction.py --task mnli
"""
import argparse

import torch.nn.functional as F
from transformers import AutoTokenizer

from uat_lite import data
from uat_lite.calibration import fit_temperature
from uat_lite.inference import UAT_ESTIMATORS, load_checkpoint, predict_deterministic, predict_mc_dropout
from uat_lite.metrics import summarize
from uat_lite.protocol import METHODS, method_notes
from uat_lite.utils import add_common_args, ckpt, save_json

THRESHOLDS = (0.9, 0.8, 0.7)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["sst2", "mnli", "squad"])
    p = add_common_args(p, None, None)
    args = p.parse_args()
    args.ckpt_path = args.ckpt_path or ckpt(args.task)
    args.output_json = args.output_json or f"./results/selective_pred_{args.task}.json"

    model = load_checkpoint(args.ckpt_path, args.device)
    tok = AutoTokenizer.from_pretrained(model.base_model_name)
    dev, test = data.dev_test_loaders(args.task, tok, args.max_length, args.batch_size)

    logits, labels = predict_deterministic(model, test, args.device)
    dev_logits, dev_labels = predict_deterministic(model, dev, args.device)
    T_base = fit_temperature(dev_logits, dev_labels)
    mc_probs, _ = predict_mc_dropout(model, test, args.M, args.device)
    uat_logits, uat_probs, _ = UAT_ESTIMATORS[args.uat](model, test, args.M, args.device)
    dev_uat_logits, _, dev_uat_labels = UAT_ESTIMATORS[args.uat](model, dev, args.M, args.device)
    T_uat = fit_temperature(dev_uat_logits, dev_uat_labels)

    probs = {"base_deterministic": F.softmax(logits, -1), "base_ts": F.softmax(logits / T_base, -1),
             "mc_global": mc_probs, "uat_lite": uat_probs, "uat_lite_ts": F.softmax(uat_logits / T_uat, -1)}
    labels = labels.numpy()
    results = {k: summarize(probs[k].numpy(), labels, THRESHOLDS) for k, _ in METHODS}
    results["base_ts"]["T"], results["uat_lite_ts"]["T"] = T_base, T_uat

    print(f"\n  Selective prediction | {args.task.upper()} (N={len(labels)})")
    print("  {:>14s}".format("Method") + "".join(f"  {'Cov@.' + t:>7s}  {'Acc@.' + t:>7s}  {'N@.' + t:>5s}"
                                                 for t in "987") + f"  {'AURC':>7s}")
    for key, label in METHODS:
        m = results[key]
        print(f"  {label:>14s}" + "".join(f"  {m['cov@' + t]:>7.4f}  {m['acc@' + t]:>7.4f}  {m['n@' + t]:>5d}"
                                          for t in ("0.9", "0.8", "0.7")) + f"  {m['aurc']:>7.4f}")

    save_json({"task": args.task, "M": args.M, "N_test": len(labels), "thresholds": list(THRESHOLDS),
               "uat": args.uat, "method_notes": method_notes(args.uat),
               "methods": results}, args.output_json)


if __name__ == "__main__":
    main()
