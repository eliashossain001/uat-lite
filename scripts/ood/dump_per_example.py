"""Save per-example predictions of the five methods on HANS / ANLI / SNLI / IMDb,
for risk-coverage curves (consumed by scripts/ood/risk_coverage.py).

Writes results/per_example/{DATASET}_{method}.npz with
labels, preds, confidences, probs, M, T, N.

    python scripts/ood/dump_per_example.py --datasets anli snli imdb hans
"""
import argparse
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from uat_lite import data
from uat_lite.inference import load_checkpoint
from uat_lite.protocol import METHODS, run_five_methods
from uat_lite.utils import add_uat_arg, ckpt, default_device

# name -> (checkpoint task, loader builder returning (test_loader, labels, dev_loader), 3->2 mapping)
DATASETS = {
    "hans": ("mnli", lambda tok, a: (*data.load_hans_tsv(tok, a.hans_tsv, a.max_length, a.batch_size)[::2],
                                     data.mnli_dev_loader_tsv(tok, a.mnli_dev_tsv, a.max_length,
                                                              a.batch_size, a.seed, 0.5)), True),
    "anli": ("mnli", lambda tok, a: (*data.load_anli(tok, a.max_length, a.batch_size)[::2],
                                     data.mnli_dev_loader(tok, a.max_length, a.batch_size)), False),
    "snli": ("mnli", lambda tok, a: (*data.load_snli_test(tok, a.max_length, a.batch_size),
                                     data.mnli_dev_loader(tok, a.max_length, a.batch_size)), False),
    "imdb": ("sst2", lambda tok, a: (*data.load_imdb_test(tok, a.max_length, a.batch_size),
                                     data.sst2_full_dev_loader(tok, a.max_length, a.batch_size)), False),
}
DISPLAY = {"hans": "HANS", "anli": "ANLI", "snli": "SNLI", "imdb": "IMDb"}


def save_npz(out_dir, dataset, key, label, labels, probs, M, T):
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs).astype(np.float64)
    path = out_dir / f"{dataset}_{key}.npz"
    np.savez_compressed(path, labels=labels, preds=probs.argmax(1).astype(np.int64),
                        confidences=probs.max(1), probs=probs, dataset=np.array(dataset),
                        method=np.array(key), method_label=np.array(label),
                        M=np.array(M, dtype=np.int64),
                        T=np.array(np.nan if T is None else T, dtype=np.float64),
                        N=np.array(len(labels), dtype=np.int64))
    print(f"    [saved] {path.name}  N={len(labels)}  acc={(probs.argmax(1) == labels).mean():.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["anli", "snli", "imdb", "hans"], choices=list(DATASETS))
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=default_device())
    p.add_argument("--hans_tsv", default="./data/hans/heuristics_evaluation_set.txt")
    p.add_argument("--mnli_dev_tsv", default="./data/mnli/dev_matched.tsv")
    p.add_argument("--out_dir", default="./results/per_example")
    add_uat_arg(p)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        t0 = time.time()
        task, build, to2 = DATASETS[name]
        print(f"\n=== {DISPLAY[name]} ({task} checkpoint) ===")
        model = load_checkpoint(ckpt(task), args.device)
        tok = AutoTokenizer.from_pretrained(model.base_model_name)
        test, labels, dev = build(tok, args)
        _, labels, probs, temps = run_five_methods(model, test, dev, args.M, args.device,
                                                   map_to_2class=to2, return_probs=True, uat=args.uat)
        for key, label in METHODS:
            M = 1 if key.startswith("base") else args.M
            save_npz(out_dir, DISPLAY[name], key, label, labels, probs[key], M, temps.get(key))
        print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
