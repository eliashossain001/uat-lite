"""Q/K/V placement ablation on the test half.

Default (``--impl attention``): UAT-LITE attention modulation (paper Eq. for u_ij) with
variants none / q / k / qk / v / qkv, all with the same M passes and dropout.
``--impl pooling`` runs the pooling-level proxy used for the submitted figure
(variants k / q / v / qkv on top of global MC dropout; "q" and "v" coincide there).

    python scripts/analysis/qkv_ablation.py --task sst2
"""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from uat_lite import data, metrics
from uat_lite.inference import load_checkpoint, predict_mc_dropout_logits, predict_pooling_proxy, predict_uat
from uat_lite.utils import ckpt, default_device, save_json, set_seed

VARIANTS = {"attention": ["none", "q", "k", "qk", "v", "qkv"], "pooling": ["original", "k", "q", "v", "qkv"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["sst2", "mnli", "squad"])
    p.add_argument("--impl", default="attention", choices=["attention", "pooling"])
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--lam", type=float, default=None, help="default: lambda stored in the checkpoint")
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=default_device())
    p.add_argument("--output_dir", default="./results/qkv_ablation")
    args = p.parse_args()

    model = load_checkpoint(args.ckpt_path or ckpt(args.task), args.device)
    lam = model.uncertainty_penalty if args.lam is None else args.lam
    _, test = data.dev_test_loaders(args.task, AutoTokenizer.from_pretrained(model.base_model_name),
                                    args.max_length, args.batch_size)
    results = {}
    for v in VARIANTS[args.impl]:
        set_seed(args.seed)  # same dropout masks for every variant
        if args.impl == "attention":
            logits, _, labels = predict_uat(model, test, args.M, args.device, lam=lam, variant=v)
        elif v == "original":
            logits, _, labels = predict_mc_dropout_logits(model, test, args.M, args.device)
        else:
            logits, labels = predict_pooling_proxy(model, test, args.M, lam, args.device, v)
        results[v] = m = metrics.from_logits(logits, labels, args.bins)
        print(f"  {v:>8s}  acc={m['accuracy']:.4f}  ece={m['ece']:.4f}  brier={m['brier']:.4f}  nll={m['nll']:.4f}")
    save_json({"task": args.task, "impl": args.impl, "M": args.M, "lambda": lam, "seed": args.seed,
               "results": results}, Path(args.output_dir) / f"{args.task}_{args.impl}_M{args.M}.json")


if __name__ == "__main__":
    main()
