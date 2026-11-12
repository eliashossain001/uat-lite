"""Sensitivity to the uncertainty penalty lambda (MNLI dev half, M fixed).

Default (``--impl attention``): UAT-LITE Q-only attention modulation; lambda = 0 is the
same MC procedure without modulation. ``--impl pooling`` runs the [CLS]-scaling proxy
used for the submitted figure (lambda = 0 there is plain MC dropout with logit averaging).

    python scripts/analysis/lambda_sweep.py
"""
import argparse

from transformers import AutoTokenizer

from uat_lite import data, metrics
from uat_lite.inference import load_checkpoint, predict_mc_dropout_logits, predict_pooling_proxy, predict_uat
from uat_lite.utils import ckpt, default_device, save_json

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", default=ckpt("mnli"))
    p.add_argument("--impl", default="attention", choices=["attention", "pooling"])
    p.add_argument("--variant", default="q", help="attention variant (q, k, qk, v, qkv)")
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--device", default=default_device())
    p.add_argument("--output_json", default="./results/lambda_sweep.json")
    args = p.parse_args()

    model = load_checkpoint(args.ckpt_path, args.device)
    dev = data.mnli_dev_loader(AutoTokenizer.from_pretrained(model.base_model_name),
                               args.max_length, args.batch_size)
    results = {}
    for lam in LAMBDAS:
        if args.impl == "attention":
            logits, _, labels = predict_uat(model, dev, args.M, args.device, lam=lam, variant=args.variant)
        elif lam == 0.0:
            logits, _, labels = predict_mc_dropout_logits(model, dev, args.M, args.device)
        else:
            logits, labels = predict_pooling_proxy(model, dev, args.M, lam, args.device, "q")
        results[str(lam)] = m = metrics.from_logits(logits, labels, args.bins)
        print(f"  lam={lam:.1f}  acc={m['accuracy']:.4f}  ece={m['ece']:.4f}  nll={m['nll']:.4f}  brier={m['brier']:.4f}")

    eces = [results[str(l)]["ece"] for l in LAMBDAS]
    accs = [results[str(l)]["accuracy"] for l in LAMBDAS]
    best = LAMBDAS[min(range(len(LAMBDAS)), key=lambda i: eces[i])]
    print(f"  best lambda by dev ECE: {best}  (ECE spread {max(eces) - min(eces):.4f})")
    save_json({"task": "mnli", "impl": args.impl, "variant": args.variant, "M": args.M, "split": "dev",
               "bins": args.bins, "lambdas": LAMBDAS, "results": results, "best_lambda_by_ece": best,
               "best_ece": min(eces), "ece_spread": max(eces) - min(eces),
               "acc_spread": max(accs) - min(accs)}, args.output_json)


if __name__ == "__main__":
    main()
