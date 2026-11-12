"""Calibration by linguistic phenomenon bucket (negation, numeric, biomedical, hedges):
ECE, Coverage@0.9, AURC and accuracy for Base+TS, UAT-LITE and UAT-LITE+TS.

Temperatures are fitted on a held-out dev split (see ``load_splits``); phenomena are
detected on the test text (premise + hypothesis for MNLI).

    python scripts/analysis/behavioral.py --dataset medqa \
        --ckpt_path ./experiments/medqa/bert-base-uncased/checkpoints/best_model.pt
"""
import argparse
import re

import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer

from uat_lite import data, metrics
from uat_lite.calibration import fit_temperature
from uat_lite.inference import UAT_ESTIMATORS, load_checkpoint, predict_deterministic
from uat_lite.utils import add_uat_arg, default_device, save_json

NEGATION_CUES = {"not", "never", "no", "without", "neither", "nor", "cannot", "don't", "doesn't",
                 "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
                 "weren't", "none", "nothing", "nowhere"}
HEDGE_CUES = {"may", "might", "could", "likely", "possibly", "probably", "suggest", "suggests",
              "suggested", "appear", "appears", "appeared", "uncertain", "unclear", "potential",
              "potentially", "approximate", "approximately", "estimated", "perhaps"}
BIOMED_SUFFIXES = ("-itis", "-osis", "-emia", "-ectomy", "-ology", "-opathy", "-plasty", "-scopy",
                   "-tomy", "-gram", "-genic", "-lysis", "-penia", "-philia", "-trophy")
NUMERIC_RE = re.compile(r"\d+")
ABBREV_RE = re.compile(r"\b[A-Z]{2,}\b")
BUCKETS = ["all", "negation", "numeric", "biomedical", "hedges"]


def detect_phenomena(text):
    tokens = text.lower().split()
    found = set()
    if NEGATION_CUES.intersection(tokens):
        found.add("negation")
    if NUMERIC_RE.search(text):
        found.add("numeric")
    if ABBREV_RE.search(text) or any(w.endswith(s) for w in tokens for s in BIOMED_SUFFIXES):
        found.add("biomedical")
    if HEDGE_CUES.intersection(tokens):
        found.add("hedges")
    return found


def load_splits(name, seed=42):
    """Returns {"dev": (texts_a, texts_b, labels), "test": (...)}; texts_b is None for single-text tasks.

    dev is held out for temperature fitting: MedQA = 10% of train, PubMedQA = the
    validation split, MNLI / SST-2 = the dev half of the validation set.
    """
    if name == "medqa":
        texts, labels = data.medqa_examples("train")
        dev_idx = np.random.RandomState(seed).permutation(len(texts))[int(len(texts) * 0.9):]
        test = data.medqa_examples("test")
        return {"dev": ([texts[i] for i in dev_idx], None, [labels[i] for i in dev_idx]),
                "test": (test[0], None, test[1])}
    if name == "pubmedqa":
        texts, labels = data.pubmedqa_examples()
        _, val_idx, test_idx = data.pubmedqa_split_indices(len(texts), seed)
        pick = lambda idx: ([texts[i] for i in idx], None, [labels[i] for i in idx])
        return {"dev": pick(val_idx), "test": pick(test_idx)}
    a, b, labels = data.load_task_texts(name)
    dev_idx, test_idx = data.dev_test_indices(len(labels), 0.5, seed)
    pick = lambda idx: ([a[i] for i in idx], None if b is None else [b[i] for i in idx], labels[idx])
    return {"dev": pick(dev_idx), "test": pick(test_idx)}


def eval_bucket(probs, labels, mask):
    if mask.sum() == 0:
        return {"n": 0, "ece": None, "cov@0.9": None, "aurc": None, "acc": None}
    p, l = probs[mask], labels[mask]
    correct = (p.argmax(1) == l).astype(float)
    return {"n": int(mask.sum()), "ece": metrics.ece(p, l),
            "cov@0.9": metrics.threshold_metrics(p.max(1), correct, 0.9)["coverage"],
            "aurc": metrics.aurc(p.max(1), correct), "acc": float(correct.mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--dataset", required=True, choices=["medqa", "pubmedqa", "mnli", "sst2"])
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default=default_device())
    p.add_argument("--output_json", default=None)
    add_uat_arg(p)
    args = p.parse_args()
    args.output_json = args.output_json or f"./results/behavioral_{args.dataset}.json"

    model = load_checkpoint(args.ckpt_path, args.device)
    tok = AutoTokenizer.from_pretrained(model.base_model_name)
    splits = load_splits(args.dataset)
    loaders = {k: data.make_loader(data.encode(tok, a, b, args.max_length), np.array(l), args.batch_size)
               for k, (a, b, l) in splits.items()}
    a, b, labels = splits["test"]
    labels = np.array(labels)
    texts = a if b is None else [f"{x} {y}" for x, y in zip(a, b)]

    found = [detect_phenomena(t) for t in texts]
    masks = {k: np.array([k in f for f in found]) for k in BUCKETS[1:]}
    masks["all"] = np.ones(len(labels), dtype=bool)

    logits, _ = predict_deterministic(model, loaders["test"], args.device)
    T_base = fit_temperature(*predict_deterministic(model, loaders["dev"], args.device))
    uat = UAT_ESTIMATORS[args.uat]
    uat_logits, uat_probs, _ = uat(model, loaders["test"], args.M, args.device)
    dev_logits, _, dev_labels = uat(model, loaders["dev"], args.M, args.device)
    T_uat = fit_temperature(dev_logits, dev_labels)
    probs = {"Base+TS": F.softmax(logits / T_base, -1).numpy(), "UAT-LITE": uat_probs.numpy(),
             "UAT-LITE+TS": F.softmax(uat_logits / T_uat, -1).numpy()}

    results = {b: {m: eval_bucket(pr, labels, masks[b]) for m, pr in probs.items()} for b in BUCKETS}
    for b in BUCKETS:
        print(f"\n  --- {b} (N={int(masks[b].sum())}) ---")
        for m, r in results[b].items():
            if r["n"]:
                print(f"  {m:>12s}  ECE={r['ece']:.4f}  Cov@0.9={r['cov@0.9']:.4f}  AURC={r['aurc']:.4f}  Acc={r['acc']:.4f}")
    save_json({"dataset": args.dataset, "uat": args.uat, "M": args.M, "T_base": T_base, "T_uat": T_uat,
               "bucket_counts": {k: int(v.sum()) for k, v in masks.items()}, "results": results},
              args.output_json)


if __name__ == "__main__":
    main()
