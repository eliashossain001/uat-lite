# -*- coding: utf-8 -*-
"""
negation_suite.py -- Negation Consistency Diagnostic for UAT-LITE (MNLI)

Evaluate how model confidence and (label-free) uncertainty signals
change under controlled negation perturbations applied to the MNLI hypothesis.

Protocol:
  1. Sample N examples from MNLI validation_matched TEST half (seeded; log indices).
  2. Create negated hypotheses using deterministic templates:
       - prefix_not_true: "It is not true that {H}"
       - prefix_false:    "It is false that {H}"
  3. Evaluate methods:
       (a) Base deterministic (single forward, dropout OFF)
       (b) UAT-LITE with MC dropout (Q-only modulation, M=10, lam=0.5)
       (c) UAT-LITE + TS (fit T on dev MC-mean logits, apply to test)
  4. Report:
       - AccOrig / AccNeg  (NOTE: AccNeg uses original labels; labels may change under negation)
       - MPOrig / MPNeg    (mean max prob)
       - MarginOrig / MarginNeg (mean p1 - p2; label-free)
       - EntOrig / EntNeg  (mean predictive entropy; label-free)
       - dConf, dMargin, dEnt, dErr

Run from repo root:
  python scripts/analysis/negation.py --device cuda
  python scripts/analysis/negation.py --device cpu

Outputs:
  - Console summary table
  - ./results/negation_consistency.json
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

# ----------------------------
# Robust repo-root import
# ----------------------------
# Assumes file lives at: <repo>/experiments/mnli/negation_suite.py

from uat_lite.calibration import fit_temperature
from uat_lite.inference import enable_dropout as _enable_mc_dropout, load_checkpoint, predict_uat
from uat_lite.models import UATLite


# ============================================================
# 1. Model loading
# ============================================================


# ============================================================
# 2. Negation templates
# ============================================================

NEGATION_TEMPLATES = {
    "prefix_not_true": "It is not true that {}",
    "prefix_false":    "It is false that {}",
}


def negate_hypothesis(hypothesis: str, template_key: str = "prefix_not_true") -> str:
    template = NEGATION_TEMPLATES[template_key]
    h = hypothesis.strip()
    return template.format(h)


# ============================================================
# 3. Dataset preparation
# ============================================================

class PairDataset(Dataset):
    def __init__(self, encodings, labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def prepare_data(
    tokenizer,
    premises,
    hypotheses,
    labels,
    max_length: int = 128,
    batch_size: int = 32,
) -> DataLoader:
    encodings = tokenizer(
        list(premises),
        list(hypotheses),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    labels_t = torch.tensor(labels, dtype=torch.long)
    ds = PairDataset(encodings, labels_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def sample_mnli(N: int = 200, seed: int = 42, dev_ratio: float = 0.5):
    """
    Load MNLI validation_matched, create a seeded permutation, split into dev/test halves,
    sample N examples from the TEST half. Also returns DEV indices for TS fitting.

    Returns dict: premises, hypotheses, labels, sample_indices (original MNLI indices),
                  val_data (HF dataset), dev_indices (indices for TS fit)
    """
    raw = load_dataset("glue", "mnli")
    val = raw["validation_matched"]

    total = len(val)
    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(total)

    n_dev = int(total * dev_ratio)
    dev_indices = all_indices[:n_dev]
    test_indices = all_indices[n_dev:]

    rng2 = np.random.RandomState(seed)
    chosen = rng2.choice(len(test_indices), size=min(N, len(test_indices)), replace=False)
    sample_idx = test_indices[chosen]

    premises = [val[int(i)]["premise"] for i in sample_idx]
    hypotheses = [val[int(i)]["hypothesis"] for i in sample_idx]
    labels = [val[int(i)]["label"] for i in sample_idx]

    return {
        "premises": premises,
        "hypotheses": hypotheses,
        "labels": labels,
        "sample_indices": sample_idx.tolist(),
        "val_data": val,
        "dev_indices": dev_indices.tolist(),
    }


# ============================================================
# 4. Inference methods
# ============================================================


@torch.no_grad()
def inference_deterministic(model: UATLite, dataloader: DataLoader, device: str = "cpu"):
    """Base deterministic: single forward, dropout OFF."""
    model.eval()
    all_probs, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        logits = model.classifier(pooled)
        probs = F.softmax(logits, dim=-1)

        all_probs.append(probs.cpu())
        all_labels.append(batch["labels"])

    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def inference_mc_q(
    model: UATLite,
    dataloader: DataLoader,
    M: int = 10,
    lam: float = 0.5,
    device: str = "cpu",
):
    """
    UAT-LITE with Q-only uncertainty modulation.
    Estimate token uncertainty via std over M hidden samples, then scale CLS vector by exp(-lam * U_cls).
    Return probs, mean_logits, labels.
    """
    model.eval()
    _enable_mc_dropout(model)

    all_probs, all_logits, all_labels = [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B = int(input_ids.size(0))
        C = int(model.num_classes)

        hidden_samples = []
        for _ in range(M):
            out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden_samples.append(out.last_hidden_state)

        hidden_stack = torch.stack(hidden_samples, dim=0)  # [M,B,seq,H]
        token_unc = hidden_stack.std(dim=0).mean(dim=-1)   # [B,seq]
        cls_unc = token_unc[:, 0]                          # [B]
        q_scale = torch.exp(-lam * cls_unc)                # [B]

        logit_accum = torch.zeros(B, C, device=device)
        for m_idx in range(M):
            h = hidden_samples[m_idx]
            pooled = h[:, 0] * q_scale.unsqueeze(-1)
            pooled = model.dropout(pooled)                # MC dropout here
            logits = model.classifier(pooled)
            logit_accum += logits

        mean_logits = logit_accum / M
        probs = F.softmax(mean_logits, dim=-1)

        all_probs.append(probs.cpu())
        all_logits.append(mean_logits.cpu())
        all_labels.append(batch["labels"])

        del hidden_samples, hidden_stack, token_unc

    return torch.cat(all_probs, dim=0), torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def inference_mc_plain(
    model: UATLite,
    dataloader: DataLoader,
    M: int = 10,
    device: str = "cpu",
):
    """Plain MC dropout (for TS fitting): mean logits over M stochastic passes."""
    model.eval()
    _enable_mc_dropout(model)

    all_logits, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B = int(input_ids.size(0))
        C = int(model.num_classes)

        logit_accum = torch.zeros(B, C, device=device)
        for _ in range(M):
            out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0]
            pooled = model.dropout(pooled)
            logits = model.classifier(pooled)
            logit_accum += logits

        all_logits.append((logit_accum / M).cpu())
        all_labels.append(batch["labels"])

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ============================================================
# 5. Temperature scaling
# ============================================================


# ============================================================
# 6. Metrics (includes label-free diagnostics)
# ============================================================

def _entropy(probs_np: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs_np, eps, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)  # natural log
    return float(ent.mean())


def _margin(probs_np: np.ndarray) -> float:
    # margin = p1 - p2 (top1 minus second best); label-free confidence sharpness
    top2 = np.partition(probs_np, -2, axis=1)[:, -2:]     # unsorted two largest
    p2 = np.minimum(top2[:, 0], top2[:, 1])
    p1 = np.maximum(top2[:, 0], top2[:, 1])
    return float((p1 - p2).mean())


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor):
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    preds = probs_np.argmax(axis=1)
    acc = float((preds == labels_np).mean())
    err = 1.0 - acc
    mp = float(probs_np.max(axis=1).mean())
    margin = _margin(probs_np)
    ent = _entropy(probs_np)

    return {
        "accuracy": acc,
        "error_rate": err,
        "mean_max_prob": mp,
        "mean_margin": margin,
        "mean_entropy": ent,
    }


def compute_deltas(orig_metrics: dict, neg_metrics: dict):
    return {
        "delta_conf": neg_metrics["mean_max_prob"] - orig_metrics["mean_max_prob"],
        "delta_err":  neg_metrics["error_rate"]    - orig_metrics["error_rate"],
        "delta_margin": neg_metrics["mean_margin"] - orig_metrics["mean_margin"],
        "delta_entropy": neg_metrics["mean_entropy"] - orig_metrics["mean_entropy"],
    }


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Negation Consistency Diagnostic (MNLI)")
    parser.add_argument("--ckpt_path", type=str,
                        default="./experiments/mnli/bert-base-uncased/checkpoints/best_model.pt")
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="./results/negation_consistency.json")

    # tokenization / dataloader knobs
    parser.add_argument("--impl", default="attention", choices=["attention", "pooling"],
                        help="attention: UAT-LITE Q-only attention for both TS fitting and evaluation; "
                             "pooling: [CLS]-scaling proxy evaluated with T fitted on plain MC dropout (submitted version)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)

    # optional: subsample TS dev set to speed up
    parser.add_argument("--ts_dev_N", type=int, default=0,
                        help="If >0, subsample this many dev examples for TS fitting (seeded). 0 = use full dev half.")
    args = parser.parse_args()

    print("=" * 78)
    if args.impl == "attention":
        global inference_mc_q, inference_mc_plain

        def inference_mc_q(model, dataloader, M=10, lam=0.5, device="cpu"):
            logits, probs, labels = predict_uat(model, dataloader, M, device, lam=lam, variant="q")
            return probs, logits, labels

        def inference_mc_plain(model, dataloader, M=10, device="cpu"):
            logits, _, labels = predict_uat(model, dataloader, M, device, lam=args.lam, variant="q")
            return logits, labels

    print(f"Negation Consistency Diagnostic | N={args.N}  M={args.M}  lam={args.lam}  seed={args.seed}")
    print("=" * 78)

    # Load model and tokenizer
    model = load_checkpoint(args.ckpt_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(model.base_model_name)
    print(f"[OK] Model loaded on {args.device}")
    print(f"[OK] ckpt_path={args.ckpt_path}")
    print(f"[OK] base_model_name={model.base_model_name}")

    # Sample data
    data = sample_mnli(N=args.N, seed=args.seed)
    premises = data["premises"]
    hypotheses_orig = data["hypotheses"]
    labels = data["labels"]
    sample_indices = data["sample_indices"]
    print(f"[OK] Sampled {len(labels)} MNLI examples (logged indices)")

    # Negate hypotheses
    templates_used = ["prefix_not_true", "prefix_false"]
    negated = {tpl: [negate_hypothesis(h, tpl) for h in hypotheses_orig] for tpl in templates_used}
    for tpl in templates_used:
        print(f"[OK] Template '{tpl}': '{negated[tpl][0][:80]}'")

    # Dataloaders
    loader_orig = prepare_data(
        tokenizer, premises, hypotheses_orig, labels,
        max_length=args.max_length, batch_size=args.batch_size
    )
    loaders_neg = {
        tpl: prepare_data(
            tokenizer, premises, negated[tpl], labels,
            max_length=args.max_length, batch_size=args.batch_size
        ) for tpl in templates_used
    }

    # ----------------------------------------------------------------
    # Fit TS on dev set using MC-mean logits
    # ----------------------------------------------------------------
    print("\n-- Fitting TS on dev set (MC-mean logits) --")
    raw_val = data["val_data"]
    dev_idx = np.array(data["dev_indices"], dtype=int)

    if args.ts_dev_N and args.ts_dev_N > 0 and args.ts_dev_N < len(dev_idx):
        rng = np.random.RandomState(args.seed)
        dev_idx = rng.choice(dev_idx, size=args.ts_dev_N, replace=False)
        dev_idx = np.sort(dev_idx)
        print(f"[OK] Subsampled TS dev set to {len(dev_idx)} examples (seed={args.seed})")

    dev_premises = [raw_val[int(i)]["premise"] for i in dev_idx]
    dev_hypotheses = [raw_val[int(i)]["hypothesis"] for i in dev_idx]
    dev_labels = [raw_val[int(i)]["label"] for i in dev_idx]

    dev_loader = prepare_data(
        tokenizer, dev_premises, dev_hypotheses, dev_labels,
        max_length=args.max_length, batch_size=args.batch_size
    )
    dev_logits, dev_labs = inference_mc_plain(model, dev_loader, M=args.M, device=args.device)
    T = fit_temperature(dev_logits, dev_labs)
    print(f"[OK] Fitted T = {T:.4f}")

    # ----------------------------------------------------------------
    # Evaluate methods x {original, negated}
    # ----------------------------------------------------------------
    results = {}
    primary_tpl = "prefix_not_true"

    # Method 1: Base deterministic
    print("\n-- Method: Base Deterministic --")
    probs_base_orig, labels_t = inference_deterministic(model, loader_orig, device=args.device)
    m_base_orig = compute_metrics(probs_base_orig, labels_t)

    probs_base_neg, _ = inference_deterministic(model, loaders_neg[primary_tpl], device=args.device)
    m_base_neg = compute_metrics(probs_base_neg, labels_t)

    d_base = compute_deltas(m_base_orig, m_base_neg)
    results["base_deterministic"] = {"original": m_base_orig, "negated": m_base_neg, "deltas": d_base}

    # Method 2: UAT-LITE (Q-only)
    print(f"\n-- Method: UAT-LITE (Q-modulation, M={args.M}) --")
    probs_q_orig, logits_q_orig, labels_t = inference_mc_q(
        model, loader_orig, M=args.M, lam=args.lam, device=args.device
    )
    m_q_orig = compute_metrics(probs_q_orig, labels_t)

    probs_q_neg, logits_q_neg, _ = inference_mc_q(
        model, loaders_neg[primary_tpl], M=args.M, lam=args.lam, device=args.device
    )
    m_q_neg = compute_metrics(probs_q_neg, labels_t)

    d_q = compute_deltas(m_q_orig, m_q_neg)
    results["uat_lite_q"] = {"original": m_q_orig, "negated": m_q_neg, "deltas": d_q}

    # Method 3: UAT-LITE + TS (applied to MC-mean logits)
    print(f"\n-- Method: UAT-LITE + TS (T={T:.4f}) --")
    probs_ts_orig = F.softmax(logits_q_orig / T, dim=-1)
    probs_ts_neg = F.softmax(logits_q_neg / T, dim=-1)

    m_ts_orig = compute_metrics(probs_ts_orig, labels_t)
    m_ts_neg = compute_metrics(probs_ts_neg, labels_t)
    d_ts = compute_deltas(m_ts_orig, m_ts_neg)

    results["uat_lite_q_ts"] = {"original": m_ts_orig, "negated": m_ts_neg, "deltas": d_ts, "T": T}

    # ----------------------------------------------------------------
    # Print summary
    # NOTE: AccNeg / dErr are computed against ORIGINAL labels (labels may change under negation).
    # Label-free signals: dConf, dMargin, dEntropy are the primary "consistency" diagnostics.
    # ----------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"Negation Consistency Summary | template='{primary_tpl}'")
    print("NOTE: AccNeg and dErr use ORIGINAL labels; negation may change the correct label on MNLI.")
    print("=" * 78)
    header = (
        f"{'Method':>18s}  {'AccO':>6s}  {'AccN':>6s}  "
        f"{'MPo':>7s}  {'MPn':>7s}  {'MarO':>7s}  {'MarN':>7s}  "
        f"{'EntO':>7s}  {'EntN':>7s}  {'dConf':>7s}  {'dMar':>7s}  {'dEnt':>7s}  {'dErr':>7s}"
    )
    print(header)
    print("-" * len(header))

    def _row(label: str, r: dict):
        o, n, d = r["original"], r["negated"], r["deltas"]
        return (
            f"{label:>18s}  {o['accuracy']:6.4f}  {n['accuracy']:6.4f}  "
            f"{o['mean_max_prob']:7.4f}  {n['mean_max_prob']:7.4f}  "
            f"{o['mean_margin']:7.4f}  {n['mean_margin']:7.4f}  "
            f"{o['mean_entropy']:7.4f}  {n['mean_entropy']:7.4f}  "
            f"{d['delta_conf']:+7.4f}  {d['delta_margin']:+7.4f}  {d['delta_entropy']:+7.4f}  {d['delta_err']:+7.4f}"
        )

    print(_row("Base Det.", results["base_deterministic"]))
    print(_row("UAT(Q)", results["uat_lite_q"]))
    print(_row("UAT(Q)+TS", results["uat_lite_q_ts"]))

    # ----------------------------------------------------------------
    # Also compute a second template quickly (Base + UAT(Q)) to show trend consistency
    # ----------------------------------------------------------------
    print("\nSecond template quick-check: 'prefix_false' (Base + UAT(Q))")
    probs_base_neg2, _ = inference_deterministic(model, loaders_neg["prefix_false"], device=args.device)
    m_base_neg2 = compute_metrics(probs_base_neg2, labels_t)
    d_base2 = compute_deltas(m_base_orig, m_base_neg2)

    probs_q_neg2, _, _ = inference_mc_q(model, loaders_neg["prefix_false"], M=args.M, lam=args.lam, device=args.device)
    m_q_neg2 = compute_metrics(probs_q_neg2, labels_t)
    d_q2 = compute_deltas(m_q_orig, m_q_neg2)

    results["base_deterministic"]["negated_false"] = m_base_neg2
    results["base_deterministic"]["deltas_false"] = d_base2
    results["uat_lite_q"]["negated_false"] = m_q_neg2
    results["uat_lite_q"]["deltas_false"] = d_q2

    # concise print
    print(f"{'Method':>18s}  {'AccN':>6s}  {'dConf':>7s}  {'dMar':>7s}  {'dEnt':>7s}  {'dErr':>7s}")
    print("-" * 56)
    print(f"{'Base Det.':>18s}  {m_base_neg2['accuracy']:6.4f}  {d_base2['delta_conf']:+7.4f}  {d_base2['delta_margin']:+7.4f}  {d_base2['delta_entropy']:+7.4f}  {d_base2['delta_err']:+7.4f}")
    print(f"{'UAT(Q)':>18s}  {m_q_neg2['accuracy']:6.4f}  {d_q2['delta_conf']:+7.4f}  {d_q2['delta_margin']:+7.4f}  {d_q2['delta_entropy']:+7.4f}  {d_q2['delta_err']:+7.4f}")

    # ----------------------------------------------------------------
    # Save JSON
    # ----------------------------------------------------------------
    out = {
        "N": args.N,
        "M": args.M,
        "lambda": args.lam,
        "seed": args.seed,
        "T": T,
        "templates_used": templates_used,
        "sample_indices": sample_indices,
        "ckpt_path": args.ckpt_path,
        "base_model_name": getattr(model, "base_model_name", None),
        "metrics_per_method": results,
        "notes": {
            "label_note": "AccNeg/dErr computed against original labels; negation may change gold label on MNLI.",
            "primary_consistency_signals": ["delta_conf", "delta_margin", "delta_entropy"],
        },
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()
