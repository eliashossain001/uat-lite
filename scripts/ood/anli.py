"""ANLI (R1-R3 test) OOD evaluation of the MNLI checkpoint, with an optional HANS
fixed-coverage companion run (HANS via HF datasets).

    python scripts/ood/anli.py [--skip_hans]
"""
import argparse

import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer

from uat_lite import data, metrics
from uat_lite.inference import load_checkpoint, predict_deterministic
from uat_lite.protocol import (method_notes, print_fixed_coverage_table, print_latex_rows,
                               print_threshold_table, run_five_methods)
from uat_lite.utils import add_common_args, ckpt, save_json

NLI_LABELS = [(0, "entailment"), (1, "neutral"), (2, "contradiction")]


def per_round(probs, labels, rounds):
    rounds = np.array(rounds)
    out = {}
    for rn in ["R1", "R2", "R3"]:
        mask = rounds == rn
        if not mask.any():
            continue
        p, l = probs[mask], labels[mask]
        correct = (p.argmax(1) == l).astype(float)
        r = {"n": int(mask.sum()), "accuracy": float(correct.mean()), "ece": metrics.ece(p, l),
             "mean_max_prob": float(p.max(1).mean()), "aurc": metrics.aurc(p.max(1), correct)}
        for lbl, name in NLI_LABELS:
            lm = l == lbl
            if lm.any():
                r[f"{name}_acc"] = float(correct[lm].mean())
                r[f"{name}_n"] = int(lm.sum())
        out[rn] = r
    return out


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__), ckpt("mnli"),
                        "./results/anli_ood.json")
    p.add_argument("--skip_hans", action="store_true")
    args = p.parse_args()

    model = load_checkpoint(args.ckpt_path, args.device)
    tok = AutoTokenizer.from_pretrained(model.base_model_name)
    dev = data.mnli_dev_loader(tok, args.max_length, args.batch_size)

    test, rounds, labels = data.load_anli(tok, args.max_length, args.batch_size)
    anli, _ = run_five_methods(model, test, dev, args.M, args.device, uat=args.uat)
    logits, _ = predict_deterministic(model, test, args.device)
    anli["base_deterministic"]["per_round"] = per_round(F.softmax(logits, -1).numpy(), labels, rounds)

    print_threshold_table("ANLI OOD -- threshold-based selective prediction", anli, len(labels))
    print_fixed_coverage_table("ANLI OOD", anli, len(labels))
    for rn, r in anli["base_deterministic"]["per_round"].items():
        print(f"  {rn}: N={r['n']:>5d}  Acc={r['accuracy']:.4f}  ECE={r['ece']:.4f}  AURC={r['aurc']:.4f}")
    print_latex_rows("ANLI", anli)

    out = {"datasets": {"anli": {"N": len(labels),
                                 "label_mapping": "3-class: entailment=0, neutral=1, contradiction=2",
                                 "methods": anli}},
           "M": args.M, "uat": args.uat, "method_notes": method_notes(args.uat),
           "fixed_coverage_note": "Acc@C%: admit top-C% by max_prob ranking (TS-invariant)"}

    if not args.skip_hans:
        test, _, hans_labels = data.load_hans_hf(tok, args.max_length, args.batch_size)
        hans, _ = run_five_methods(model, test, dev, args.M, args.device, map_to_2class=True, uat=args.uat)
        print_threshold_table("HANS OOD -- threshold-based selective prediction", hans, len(hans_labels))
        print_fixed_coverage_table("HANS OOD", hans, len(hans_labels))
        out["datasets"]["hans"] = {
            "N": len(hans_labels),
            "label_mapping": "2-class: entailment=0, non_entailment=1; neutral+contradiction merged",
            "methods": hans}
    save_json(out, args.output_json)


if __name__ == "__main__":
    main()
