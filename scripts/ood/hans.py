"""HANS OOD evaluation of the MNLI checkpoint (paper: OOD selective prediction, HANS).

HANS is read from the official TSV (downloaded if missing). Temperatures are fitted
on the dev half of the GLUE MNLI dev_matched.tsv if present (as in the paper),
otherwise on the dev half of the HF MNLI validation_matched split.
3-class predictions are mapped to (entailment, non-entailment) after softmax.

    python scripts/ood/hans.py --mnli_dev_tsv ./data/mnli/dev_matched.tsv
"""
import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from uat_lite import data
from uat_lite.inference import load_checkpoint
from uat_lite.protocol import method_notes, METHODS, print_threshold_table, run_five_methods
from uat_lite.utils import add_common_args, ckpt, save_json

HEURISTICS = ["lexical_overlap", "subsequence", "constituent"]


def per_heuristic(probs, labels, heuristics):
    heuristics = np.array(heuristics)
    out = {}
    for h in HEURISTICS:
        mask = heuristics == h
        if not mask.any():
            continue
        p, l = probs[mask], labels[mask]
        correct = (p.argmax(1) == l).astype(float)
        r = {"n": int(mask.sum()), "accuracy": float(correct.mean()),
             "mean_max_prob": float(p.max(1).mean())}
        for lbl, name in [(0, "entailment"), (1, "non_entailment")]:
            lm = l == lbl
            if lm.any():
                r[f"{name}_acc"] = float(correct[lm].mean())
                r[f"{name}_n"] = int(lm.sum())
                r[f"{name}_mean_conf"] = float(p[lm].max(1).mean())
        out[h] = r
    return out


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__), ckpt("mnli"),
                        "./results/hans_ood.json")
    p.add_argument("--hans_tsv", default="./data/hans/heuristics_evaluation_set.txt")
    p.add_argument("--mnli_dev_tsv", default="./data/mnli/dev_matched.tsv")
    p.add_argument("--mnli_dev_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model = load_checkpoint(args.ckpt_path, args.device)
    tok = AutoTokenizer.from_pretrained(model.base_model_name)
    test, heuristics, labels = data.load_hans_tsv(tok, args.hans_tsv, args.max_length, args.batch_size)
    if Path(args.mnli_dev_tsv).exists():
        dev = data.mnli_dev_loader_tsv(tok, args.mnli_dev_tsv, args.max_length, args.batch_size,
                                       args.seed, args.mnli_dev_frac)
    else:  # the submitted numbers used the GLUE TSV; the HF split differs slightly
        print(f"[data] {args.mnli_dev_tsv} not found; fitting temperatures on the HF MNLI dev half")
        dev = data.mnli_dev_loader(tok, args.max_length, args.batch_size, args.seed)

    results, labels, probs, _ = run_five_methods(model, test, dev, args.M, args.device,
                                                 map_to_2class=True, return_probs=True, uat=args.uat)
    for key, _ in METHODS:
        results[key]["per_heuristic"] = per_heuristic(probs[key], labels, heuristics)

    print_threshold_table("HANS OOD -- threshold-based selective prediction", results, len(labels))
    print("\n  {:>14s}".format("Method") + "".join(f"  {h:>16s}" for h in HEURISTICS) + "   (all / non-entailment)")
    for key, label in METHODS:
        ph = results[key]["per_heuristic"]
        print(f"  {label:>14s}" + "".join(
            f"  {ph[h]['accuracy']:.4f} / {ph[h].get('non_entailment_acc', float('nan')):.4f}" for h in HEURISTICS))

    save_json({
        "dataset": "HANS", "M": args.M, "N": len(labels),
        "label_mapping": "entailment=0, non_entailment=1; MNLI neutral+contradiction merged",
        "data_paths": {"hans_tsv": args.hans_tsv, "mnli_dev_tsv": args.mnli_dev_tsv},
        "uat": args.uat, "method_notes": method_notes(args.uat), "methods": results,
    }, args.output_json)


if __name__ == "__main__":
    main()
