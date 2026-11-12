# -*- coding: utf-8 -*-
"""
negation_suite_extended.py -- Negation + Calibration Sanity Checks for UAT-LITE

Extends the base negation diagnostic with two sanity checks:

  (1) LayerNorm-invariance: compute U_LN(x_j) by applying LayerNorm to
      per-pass hidden states before computing std across passes.
      Show ECE + negation trends are unchanged.

  (2) Alternative uncertainty aggregation:
      --uncert_variant meanstd  (default: mean of per-dim std)
      --uncert_variant rmsstd   (RMS of per-dim std)
      --uncert_variant medianstd (median of per-dim std)

Runs all combinations automatically and saves a combined JSON.

Usage:
  python scripts/analysis/negation_sanity.py --device cuda
  python scripts/analysis/negation_sanity.py --device cpu --M 10 --lam 0.5

Outputs:
  - Console comparison table
  - ./results/negation_sanity_checks.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from uat_lite.inference import enable_dropout as _enable_mc_dropout, load_checkpoint
from uat_lite.metrics import ece as compute_ece


# ============================================================
# 1. Model loading
# ============================================================


# ============================================================
# 2. Token uncertainty variants
# ============================================================

def compute_token_uncertainty(hidden_stack, variant="meanstd", use_ln=False):
    """
    Compute per-token uncertainty from MC hidden state samples.

    Args:
        hidden_stack: [M, B, seq, hidden] tensor
        variant: one of "meanstd", "rmsstd", "medianstd"
        use_ln: if True, apply LayerNorm per token per pass before std

    Returns:
        token_unc: [B, seq] tensor
    """
    M, B, S, H = hidden_stack.shape

    if use_ln:
        # Apply LayerNorm across hidden dim for each token in each pass
        # This tests whether the uncertainty signal is invariant to
        # per-token scale/shift normalization
        ln = nn.LayerNorm(H, elementwise_affine=False).to(hidden_stack.device)
        # Reshape to apply LN: [M*B*S, H]
        flat = hidden_stack.reshape(-1, H)
        normed = ln(flat)
        hidden_proc = normed.reshape(M, B, S, H)
    else:
        hidden_proc = hidden_stack

    # Per-dimension std across M passes: [B, seq, hidden]
    per_dim_std = hidden_proc.std(dim=0)

    if variant == "meanstd":
        # Mean of per-dim std -> [B, seq]
        token_unc = per_dim_std.mean(dim=-1)
    elif variant == "rmsstd":
        # RMS of per-dim std -> [B, seq]
        token_unc = (per_dim_std ** 2).mean(dim=-1).sqrt()
    elif variant == "medianstd":
        # Median of per-dim std -> [B, seq]
        token_unc = per_dim_std.median(dim=-1).values
    else:
        raise ValueError("Unknown variant: {}".format(variant))

    return token_unc


# ============================================================
# 3. Data preparation
# ============================================================

NEGATION_TEMPLATES = {
    "prefix_not_true": "It is not true that {}",
}


def negate_hypothesis(hypothesis, template_key="prefix_not_true"):
    template = NEGATION_TEMPLATES[template_key]
    h = hypothesis.strip()
    if h and h[0].isupper():
        h = h[0].lower() + h[1:]
    return template.format(h)


class PairDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def prepare_data(tokenizer, premises, hypotheses, labels, max_length=128):
    encodings = tokenizer(
        list(premises), list(hypotheses),
        truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )
    labels_t = torch.tensor(labels, dtype=torch.long)
    ds = PairDataset(encodings, labels_t)
    return DataLoader(ds, batch_size=32, shuffle=False)


def sample_mnli(N=200, seed=42, dev_ratio=0.5):
    raw = load_dataset("glue", "mnli")
    val = raw["validation_matched"]

    total = len(val)
    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(total)
    n_dev = int(total * dev_ratio)
    test_indices = all_indices[n_dev:]

    rng2 = np.random.RandomState(seed)
    chosen = rng2.choice(len(test_indices), size=min(N, len(test_indices)), replace=False)
    sample_idx = test_indices[chosen]

    premises = [val[int(i)]["premise"] for i in sample_idx]
    hypotheses = [val[int(i)]["hypothesis"] for i in sample_idx]
    labels = [val[int(i)]["label"] for i in sample_idx]

    # Dev half for ECE computation
    dev_indices = all_indices[:n_dev]

    return {
        "premises": premises,
        "hypotheses": hypotheses,
        "labels": labels,
        "sample_indices": sample_idx.tolist(),
        "val_data": val,
        "dev_indices": dev_indices,
    }


# ============================================================
# 4. Inference with configurable uncertainty
# ============================================================


def inference_deterministic(model, dataloader, device="cpu"):
    """Base deterministic: dropout OFF."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0]
            logits = model.classifier(pooled)
            all_probs.append(F.softmax(logits, dim=-1).cpu())
            all_labels.append(batch["labels"])
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def inference_mc_q_configurable(model, dataloader, M, lam, device,
                                 uncert_variant="meanstd", use_ln=False):
    """
    Q-only uncertainty modulation with configurable uncertainty computation.
    """
    model.eval()
    _enable_mc_dropout(model)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader,
                          desc="MC Q var={} ln={}".format(uncert_variant, use_ln),
                          leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            B = input_ids.size(0)
            C = model.num_classes

            # Collect hidden states
            hidden_samples = []
            for _ in range(M):
                out = model.transformer(
                    input_ids=input_ids, attention_mask=attention_mask)
                hidden_samples.append(out.last_hidden_state)

            hidden_stack = torch.stack(hidden_samples, dim=0)  # [M, B, seq, H]

            # Compute token uncertainty with chosen variant + LN option
            token_unc = compute_token_uncertainty(
                hidden_stack, variant=uncert_variant, use_ln=use_ln)

            cls_unc = token_unc[:, 0]  # [B]
            q_scale = torch.exp(-lam * cls_unc)  # [B]

            logit_accum = torch.zeros(B, C, device=device)
            for m_idx in range(M):
                h = hidden_samples[m_idx]
                pooled = h[:, 0] * q_scale.unsqueeze(-1)
                pooled = model.dropout(pooled)
                logits = model.classifier(pooled)
                logit_accum += logits

            mean_logits = logit_accum / M
            probs = F.softmax(mean_logits, dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(batch["labels"])

            del hidden_samples, hidden_stack, token_unc

    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


# ============================================================
# 5. Metrics
# ============================================================


def compute_all_metrics(probs_t, labels_t):
    probs_np = probs_t.numpy()
    labels_np = labels_t.numpy()
    preds = probs_np.argmax(axis=1)
    acc = float((preds == labels_np).mean())
    ece = compute_ece(probs_np, labels_np)
    mean_maxprob = float(probs_np.max(axis=1).mean())
    return {
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "ece": ece,
        "mean_max_prob": mean_maxprob,
    }


# ============================================================
# 6. Dev set ECE evaluation
# ============================================================

def eval_dev_ece(model, tokenizer, data, M, lam, device,
                 uncert_variant, use_ln):
    """Run Q-variant on full dev set, return ECE."""
    val_data = data["val_data"]
    dev_idx = data["dev_indices"]
    dev_premises = [val_data[int(i)]["premise"] for i in dev_idx]
    dev_hypotheses = [val_data[int(i)]["hypothesis"] for i in dev_idx]
    dev_labels = [val_data[int(i)]["label"] for i in dev_idx]
    dev_loader = prepare_data(tokenizer, dev_premises, dev_hypotheses, dev_labels)

    probs, labels = inference_mc_q_configurable(
        model, dev_loader, M, lam, device, uncert_variant, use_ln)
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    preds = probs_np.argmax(axis=1)
    acc = float((preds == labels_np).mean())
    ece = compute_ece(probs_np, labels_np)
    return {"dev_accuracy": acc, "dev_ece": ece}


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Negation + Calibration Sanity Checks")
    parser.add_argument("--ckpt_path", type=str,
        default="./experiments/mnli/bert-base-uncased/checkpoints/best_model.pt")
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str,
        default="./results/negation_sanity_checks.json")
    args = parser.parse_args()

    # All configurations to test
    CONFIGS = [
        {"variant": "meanstd",   "use_ln": False, "label": "meanstd"},
        {"variant": "meanstd",   "use_ln": True,  "label": "meanstd+LN"},
        {"variant": "rmsstd",    "use_ln": False, "label": "rmsstd"},
        {"variant": "rmsstd",    "use_ln": True,  "label": "rmsstd+LN"},
        {"variant": "medianstd", "use_ln": False, "label": "medianstd"},
        {"variant": "medianstd", "use_ln": True,  "label": "medianstd+LN"},
    ]

    print("=" * 75)
    print("  Negation + Calibration Sanity Checks")
    print("  N={}  M={}  lam={}  seed={}  configs={}".format(
        args.N, args.M, args.lam, args.seed, len(CONFIGS)))
    print("=" * 75)

    model = load_checkpoint(args.ckpt_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(model.base_model_name)

    # Sample data
    data = sample_mnli(N=args.N, seed=args.seed)
    premises = data["premises"]
    hypotheses_orig = data["hypotheses"]
    labels = data["labels"]
    hypotheses_neg = [negate_hypothesis(h) for h in hypotheses_orig]

    loader_orig = prepare_data(tokenizer, premises, hypotheses_orig, labels)
    loader_neg = prepare_data(tokenizer, premises, hypotheses_neg, labels)

    print("[OK] Sampled {} examples".format(len(labels)))
    print("[OK] Negated example: '{}'".format(hypotheses_neg[0][:80]))

    # ---- Base deterministic (reference) ----
    print("")
    print("-- Base Deterministic (reference) --")
    probs_base_orig, labels_t = inference_deterministic(
        model, loader_orig, device=args.device)
    probs_base_neg, _ = inference_deterministic(
        model, loader_neg, device=args.device)

    base_orig = compute_all_metrics(probs_base_orig, labels_t)
    base_neg = compute_all_metrics(probs_base_neg, labels_t)
    base_delta_conf = base_neg["mean_max_prob"] - base_orig["mean_max_prob"]
    base_delta_err = base_neg["error_rate"] - base_orig["error_rate"]

    results = {
        "base_deterministic": {
            "original": base_orig,
            "negated": base_neg,
            "delta_conf": base_delta_conf,
            "delta_err": base_delta_err,
        }
    }

    print("  Orig acc={:.4f}  Neg acc={:.4f}  dConf={:+.4f}  dErr={:+.4f}".format(
        base_orig["accuracy"], base_neg["accuracy"],
        base_delta_conf, base_delta_err))

    # ---- Each uncertainty config ----
    for cfg in CONFIGS:
        label = cfg["label"]
        print("")
        print("-- UAT-LITE(Q) [{}] --".format(label))

        # Negation diagnostic on sample
        probs_orig, _ = inference_mc_q_configurable(
            model, loader_orig, args.M, args.lam, args.device,
            cfg["variant"], cfg["use_ln"])
        probs_neg, _ = inference_mc_q_configurable(
            model, loader_neg, args.M, args.lam, args.device,
            cfg["variant"], cfg["use_ln"])

        m_orig = compute_all_metrics(probs_orig, labels_t)
        m_neg = compute_all_metrics(probs_neg, labels_t)
        delta_conf = m_neg["mean_max_prob"] - m_orig["mean_max_prob"]
        delta_err = m_neg["error_rate"] - m_orig["error_rate"]

        # Dev set ECE
        dev_metrics = eval_dev_ece(
            model, tokenizer, data, args.M, args.lam, args.device,
            cfg["variant"], cfg["use_ln"])

        results[label] = {
            "variant": cfg["variant"],
            "use_ln": cfg["use_ln"],
            "original": m_orig,
            "negated": m_neg,
            "delta_conf": delta_conf,
            "delta_err": delta_err,
            "dev_ece": dev_metrics["dev_ece"],
            "dev_accuracy": dev_metrics["dev_accuracy"],
        }

        print("  Orig acc={:.4f}  Neg acc={:.4f}  dConf={:+.4f}  dErr={:+.4f}  devECE={:.4f}".format(
            m_orig["accuracy"], m_neg["accuracy"],
            delta_conf, delta_err, dev_metrics["dev_ece"]))

    # ================================================================
    # Summary tables
    # ================================================================
    print("")
    print("=" * 75)
    print("  SUMMARY: Uncertainty Aggregation Comparison (no LN)")
    print("=" * 75)
    print("  {:>12s}  {:>7s}  {:>7s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
        "Variant", "AccOrig", "AccNeg", "dConf", "dErr", "SmpECE", "DevECE"))
    print("  " + "-" * 68)

    # Base row
    print("  {:>12s}  {:>7.4f}  {:>7.4f}  {:>+8.4f}  {:>+8.4f}  {:>8.4f}  {:>8s}".format(
        "Base Det.",
        base_orig["accuracy"], base_neg["accuracy"],
        base_delta_conf, base_delta_err,
        base_orig["ece"], "--"))

    for var_name in ["meanstd", "rmsstd", "medianstd"]:
        r = results[var_name]
        print("  {:>12s}  {:>7.4f}  {:>7.4f}  {:>+8.4f}  {:>+8.4f}  {:>8.4f}  {:>8.4f}".format(
            var_name,
            r["original"]["accuracy"], r["negated"]["accuracy"],
            r["delta_conf"], r["delta_err"],
            r["original"]["ece"], r["dev_ece"]))

    print("")
    print("=" * 75)
    print("  SUMMARY: LayerNorm Invariance (each variant +/- LN)")
    print("=" * 75)
    print("  {:>12s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
        "Variant", "dConf", "dConf+LN", "dErr", "dErr+LN", "DevECE", "DevECE+LN"))
    print("  " + "-" * 72)

    for var_name in ["meanstd", "rmsstd", "medianstd"]:
        r_no = results[var_name]
        r_ln = results[var_name + "+LN"]
        print("  {:>12s}  {:>+8.4f}  {:>+8.4f}  {:>+8.4f}  {:>+8.4f}  {:>8.4f}  {:>8.4f}".format(
            var_name,
            r_no["delta_conf"], r_ln["delta_conf"],
            r_no["delta_err"], r_ln["delta_err"],
            r_no["dev_ece"], r_ln["dev_ece"]))

    # ================================================================
    # Save JSON
    # ================================================================
    out = {
        "N": args.N,
        "M": args.M,
        "lambda": args.lam,
        "seed": args.seed,
        "negation_template": "prefix_not_true",
        "sample_indices": data["sample_indices"],
        "configs_tested": [c["label"] for c in CONFIGS],
        "results": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print("")
    print("[OK] Results saved to {}".format(args.output_json))


if __name__ == "__main__":
    main()