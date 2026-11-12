"""Penalty functional-form ablation inside UAT-LITE attention: scores are attenuated by
f(u) = exp(-lambda u) (exponential), max(0, 1 - lambda u) (linear) or 1 / (1 + lambda u)
(reciprocal). Same checkpoint, test split, M, lambda, variant and dropout masks.

    python scripts/analysis/penalty_form.py --task squad --lam 1.0
"""
import argparse

from transformers import AutoTokenizer

from uat_lite import data, metrics
from uat_lite.inference import load_checkpoint, predict_uat
from uat_lite.models.attention import PENALTIES
from uat_lite.utils import ckpt, default_device, save_json, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="squad", choices=["sst2", "mnli", "squad"])
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--variant", default="q")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=default_device())
    p.add_argument("--output_json", default=None)
    args = p.parse_args()
    args.output_json = args.output_json or f"./results/penalty_form_{args.task}.json"

    model = load_checkpoint(args.ckpt_path or ckpt(args.task), args.device)
    _, test = data.dev_test_loaders(args.task, AutoTokenizer.from_pretrained(model.base_model_name),
                                    args.max_length, args.batch_size)
    results = {}
    for name in PENALTIES:
        set_seed(args.seed)
        logits, probs, labels = predict_uat(model, test, args.M, args.device, lam=args.lam,
                                            variant=args.variant, penalty=name)
        r = metrics.from_logits(logits, labels)
        probs, correct = probs.numpy(), (probs.argmax(1) == labels).numpy().astype(float)
        r["aurc"] = metrics.aurc(probs.max(1), correct)
        r["cov@0.9"] = metrics.threshold_metrics(probs.max(1), correct, 0.9)["coverage"]
        results[name] = r
        print(f"  {name:>12s}  " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))
    save_json({"task": args.task, "variant": args.variant, "lam": args.lam, "M": args.M, "seed": args.seed,
               "results": results}, args.output_json)


if __name__ == "__main__":
    main()
