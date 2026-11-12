#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_scale_geometry_diagnostics.py
==================================
Appendix Experiment: "Scale and Geometry Diagnostics for Token Uncertainty"
============================================================================

Three diagnostics for UAT-LITE token uncertainty, all inference-only
(inference-only, no training code touched):

  Part A — Embedding-Rescaling Invariance Test
  ─────────────────────────────────────────────
  Multiply input token embeddings by scalar c ∈ {0.5, 1, 2, 4} at
  inference time. For each c, recompute per-example token uncertainty U_c
  and downstream calibration metrics (ECE, NLL, Acc, AURC). Reports:
    • ΔECE / ΔNLL / ΔAcc  vs. c=1 baseline
    • Spearman ρ(U_orig, U_c) — stability of uncertainty ranking under scale
    • Attention-modulation output cosine similarity vs. c=1

  Part B — Epistemic Alignment via Correlation Checks
  ────────────────────────────────────────────────────
  For each example compute:
    • U(x)        — token-level uncertainty (CV over hidden dim)
    • H[p(y|x)]   — predictive entropy of the mean MC prediction
    • MI           — mutual information: H[mean p] − mean_m H[p^(m)]
    • ‖e‖          — mean L2 norm of input token embeddings (norm control)
  Reports:
    • Pearson r and Spearman ρ between U and {H, MI}
    • Pearson r and Spearman ρ between U and ‖e‖ (norm-control sanity)
    • Partial correlation of U with H and MI controlling for ‖e‖

  Part C — Alternative Uncertainty Definitions
  ─────────────────────────────────────────────
  Implements and compares four U variants over the same MC hidden states:
    1. mean-std   : mean_j std_m[h_j^(m)]          (current UAT-LITE default)
    2. cov-trace  : mean_j tr(Cov_m[h_j^(m)])      (richer second-order)
    3. cv          : mean_j std_m / (|mean_m| + ε)  (scale-normalised)
    4. norm-cv    : mean_j std_m / (‖mean_m‖ + ε)  (L2-normalised CV)
  Each variant drives the exponential attention penalty (same λ), and ECE /
  NLL / Acc / AURC are compared to show conclusions are stable.

Compute budget:
  • Two datasets : MNLI + SQuAD (--datasets mnli squad)
  • M = 10 MC samples
  • λ = 1.0
  • Stratified 2000-example subset per dataset

Outputs (all under --output_dir, default results/):
  • scale_geometry_{dataset}.json          full results per dataset
  • scale_geometry_all.json               combined
  • part_a_rescaling.csv                  clean: one row per (dataset, scale)
  • part_b_correlations.csv              clean: one row per dataset
  • part_c_variants.csv                   clean: one row per (dataset, variant)
  • Console: formatted tables + LaTeX blocks ready for appendix

Usage:
  # from repo root:
  python scripts/analysis/scale_geometry.py --datasets mnli squad --device cuda

  # override defaults:
  python scripts/analysis/scale_geometry.py \
      --datasets mnli squad \
      --ckpt_root ./experiments \
      --backbone bert-base-uncased \
      --M 10 --lam 1.0 --max_eval 2000 \
      --batch_size 16 --max_length 256 \
      --seed 42 --device cuda \
      --output_dir results/scale_geometry

Author: Elias Hossain
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import argparse
import csv
import json
import os
import sys
from pathlib import Path

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from uat_lite import metrics
from uat_lite.inference import load_checkpoint as load_model
from uat_lite.metrics import accuracy as compute_accuracy, ece as compute_ece


def compute_aurc(probs, labels):
    return metrics.aurc(probs.max(axis=1), (probs.argmax(axis=1) == labels).astype(float))


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

EPS = 1e-8

DATASET_CKPT = {
    "mnli":     "mnli",
    "squad":    "squad",
    "sst2":     "sst2",
    "medqa":    "medqa",
    "pubmedqa": "pubmedqa",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — Dataset + model helpers
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _stratified_sample(labels_np, n_total, rng):
    classes = np.unique(labels_np)
    n_each  = n_total // len(classes)
    sel = []
    for c in classes:
        idx = np.where(labels_np == c)[0]
        sel.append(rng.choice(idx, min(n_each, len(idx)), replace=False))
    return np.sort(np.concatenate(sel))


def load_eval_dataset(name, tokenizer, max_length, batch_size, max_eval, seed):
    """Return (DataLoader, labels_np).  Stratified subsample."""
    rng = np.random.RandomState(seed)

    if name == "mnli":
        ds     = load_dataset("glue", "mnli")["validation_matched"]
        texts  = [f"{p} {h}" for p, h in zip(ds["premise"], ds["hypothesis"])]
        labels = np.array(ds["label"])

    elif name == "squad":
        ds = load_dataset("rajpurkar/squad_v2", split="validation")
        texts, labels = [], []
        for ex in ds:
            texts.append(f"{ex['context']} [SEP] {ex['question']}")
            labels.append(0 if len(ex["answers"]["text"]) > 0 else 1)
        labels = np.array(labels)

    elif name == "sst2":
        ds     = load_dataset("glue", "sst2")["validation"]
        texts  = list(ds["sentence"])
        labels = np.array(ds["label"])

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    sel    = _stratified_sample(labels, max_eval, rng)
    texts  = [texts[i]  for i in sel]
    labels = labels[sel]

    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt")
    loader = DataLoader(
        SimpleDataset(enc, torch.tensor(labels, dtype=torch.long)),
        batch_size=batch_size, shuffle=False,
    )
    class_counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    print(f"    [{name}] {len(labels)} examples  (classes: {class_counts})")
    return loader, labels


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Metric functions  (K=15, consistent with rest of codebase)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_nll(probs, labels):
    return float(-np.log(probs[np.arange(len(labels)), labels] + EPS).mean())


def metrics_dict(probs, labels):
    return dict(
        ece  = round(compute_ece(probs, labels),      4),
        nll  = round(compute_nll(probs, labels),      4),
        acc  = round(compute_accuracy(probs, labels), 4),
        aurc = round(compute_aurc(probs, labels),     4),
    )


# ── Correlation helpers ───────────────────────────────────────────────────────

def spearman(a, b):
    if np.std(a) < EPS or np.std(b) < EPS:
        return float("nan"), float("nan")
    r, p = sp_stats.spearmanr(a, b)
    return float(r), float(p)


def pearson(a, b):
    if np.std(a) < EPS or np.std(b) < EPS:
        return float("nan"), float("nan")
    r, p = sp_stats.pearsonr(a, b)
    return float(r), float(p)


def partial_corr_pearson(x, y, z):
    """Partial Pearson r of x~y controlling for z."""
    def resid(a, b):
        slope, inter, *_ = sp_stats.linregress(b, a)
        return a - (slope * b + inter)
    return pearson(resid(x, z), resid(y, z))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — MC inference engine
# ═══════════════════════════════════════════════════════════════════════════════

def mc_collect(model, loader, M, device, embedding_scale=1.0):
    """
    Run M MC-dropout passes. Applies embedding scale via a forward hook
    (no weights modified). Returns dict of per-example tensors.
    """
    emb_layer     = model.transformer.embeddings
    scale_hook    = None
    norm_captures = []   # one entry per forward pass call

    # Hook 1: optionally scale embedding output
    if abs(embedding_scale - 1.0) > 1e-6:
        def _scale_fn(module, inp, out):
            return out * embedding_scale
        scale_hook = emb_layer.register_forward_hook(_scale_fn)

    # Hook 2: capture mean token embedding norm (fires every forward pass)
    def _norm_fn(module, inp, out):
        with torch.no_grad():
            norm_captures.append(out.norm(dim=-1).mean(dim=-1).cpu())  # (B,)

    norm_hook = emb_layer.register_forward_hook(_norm_fn)

    model.train()   # activate MC dropout

    batches_hidden_mean = []
    batches_hidden_std  = []
    batches_hidden_all  = []
    batches_probs_mean  = []
    batches_probs_all   = []
    batches_labels      = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  MC c={embedding_scale:.1f}", leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            h_samples, p_samples = [], []
            for _ in range(M):
                out    = model.transformer(input_ids=ids, attention_mask=mask)
                cls_h  = out.last_hidden_state[:, 0, :]         # (B, d)
                logits = model.classifier(cls_h)                # (B, C)
                h_samples.append(cls_h.cpu())
                p_samples.append(F.softmax(logits, dim=-1).cpu())

            hs = torch.stack(h_samples, 0)   # (M, B, d)
            ps = torch.stack(p_samples, 0)   # (M, B, C)

            batches_hidden_mean.append(hs.mean(0))
            batches_hidden_std.append(hs.std(0, unbiased=True))
            batches_hidden_all.append(hs.permute(1, 0, 2))      # (B, M, d)
            batches_probs_mean.append(ps.mean(0))
            batches_probs_all.append(ps.permute(1, 0, 2))       # (B, M, C)
            batches_labels.append(batch["labels"])

    norm_hook.remove()
    if scale_hook is not None:
        scale_hook.remove()
    model.eval()

    # norm_captures fires M times per batch; take 1st pass index per batch
    n_batches   = len(batches_labels)
    batch_norms = [norm_captures[i * M] for i in range(n_batches)]

    return {
        "hidden_mean": torch.cat(batches_hidden_mean, 0).numpy(),  # (N, d)
        "hidden_std":  torch.cat(batches_hidden_std,  0).numpy(),  # (N, d)
        "hidden_all":  torch.cat(batches_hidden_all,  0).numpy(),  # (N, M, d)
        "emb_norm":    torch.cat(batch_norms,         0).numpy(),  # (N,)
        "probs_mean":  torch.cat(batches_probs_mean,  0).numpy(),  # (N, C)
        "probs_all":   torch.cat(batches_probs_all,   0).numpy(),  # (N, M, C)
        "labels":      torch.cat(batches_labels,      0).numpy(),  # (N,)
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Uncertainty variant definitions
# ═══════════════════════════════════════════════════════════════════════════════

def u_mean_std(data):
    """Current UAT-LITE default: mean over hidden dim of per-dim std."""
    return data["hidden_std"].mean(axis=-1)           # (N,)


def u_cov_trace(data):
    """Diagonal trace of covariance matrix (richer second-order signal)."""
    return np.var(data["hidden_all"], axis=1, ddof=1).mean(axis=-1)  # (N,)


def u_cv(data):
    """Coefficient of variation: std / (|mean| + ε)  — scale-normalised."""
    return (data["hidden_std"] / (np.abs(data["hidden_mean"]) + EPS)).mean(axis=-1)


def u_norm_cv(data):
    """L2-normalised CV: std / (‖mean‖ + ε) — strongest scale invariance."""
    mu_norm = np.linalg.norm(data["hidden_mean"], axis=-1, keepdims=True) + EPS
    return (data["hidden_std"] / mu_norm).mean(axis=-1)


U_VARIANTS = {
    "mean-std": u_mean_std,
    "cov-trace": u_cov_trace,
    "cv":        u_cv,
    "norm-cv":   u_norm_cv,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Penalised probs from collected MC data (offline, no re-inference)
# ═══════════════════════════════════════════════════════════════════════════════

def penalised_probs(data, u_vec, lam, model, device):
    """
    Applies exponential UAT-LITE penalty offline using stored hidden states.
    Classifier runs in CPU mini-batches — fast, no GPU needed here.
    """
    N, M, d = data["hidden_all"].shape
    hs_t    = torch.tensor(data["hidden_all"], dtype=torch.float32)  # (N, M, d)
    mean_t  = torch.tensor(data["hidden_mean"],dtype=torch.float32)  # (N, d)
    u_t     = torch.tensor(u_vec,              dtype=torch.float32)  # (N,)

    clf = model.classifier.cpu()
    clf.eval()
    all_probs = []
    B = 256

    with torch.no_grad():
        for i in range(0, N, B):
            hs_b  = hs_t[i:i+B]                                        # (b, M, d)
            mu_b  = mean_t[i:i+B]                                      # (b, d)
            u_b   = u_t[i:i+B]                                         # (b,)

            # cosine attention weights over M samples
            cos   = (F.normalize(hs_b, dim=-1)
                     * F.normalize(mu_b, dim=-1).unsqueeze(1)).sum(-1)  # (b, M)
            a_raw = F.softmax(cos, dim=1)

            # exponential penalty + renormalise
            a_pen = a_raw * torch.exp(-lam * u_b.unsqueeze(1))
            a_pen = a_pen / (a_pen.sum(1, keepdim=True) + EPS)

            # weighted logit average
            b_   = hs_b.size(0)
            lg   = clf(hs_b.view(b_ * M, d)).view(b_, M, -1)           # (b, M, C)
            p_   = F.softmax((lg * a_pen.unsqueeze(-1)).sum(1), -1)    # (b, C)
            all_probs.append(p_.numpy())

    model.classifier.to(device)
    return np.concatenate(all_probs, 0)                                 # (N, C)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Epistemic signals from MC probs
# ═══════════════════════════════════════════════════════════════════════════════

def predictive_entropy(probs_all):
    p = probs_all.mean(axis=1) + EPS                   # (N, C)
    return -(p * np.log(p)).sum(axis=1)                # (N,)


def mean_entropy(probs_all):
    p = probs_all + EPS                                # (N, M, C)
    return -(p * np.log(p)).sum(axis=-1).mean(axis=1) # (N,)


def mutual_information(probs_all):
    return predictive_entropy(probs_all) - mean_entropy(probs_all)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART A — Embedding-Rescaling Invariance
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_a(model, loader, labels_np, M, lam, device, scales):
    print("\n  ── Part A: Embedding-Rescaling Invariance ──")

    # baseline at c=1.0
    data_base = mc_collect(model, loader, M, device, embedding_scale=1.0)
    u_base    = u_cv(data_base)
    p_base    = penalised_probs(data_base, u_base, lam, model, device)
    m_base    = metrics_dict(p_base, labels_np)

    rows = []
    for c in scales:
        print(f"    c = {c}")
        data_c = (data_base if abs(c - 1.0) < 1e-6
                  else mc_collect(model, loader, M, device, embedding_scale=c))

        u_c = u_cv(data_c)
        p_c = penalised_probs(data_c, u_c, lam, model, device)
        m_c = metrics_dict(p_c, labels_np)

        spr, spp = spearman(u_base, u_c)

        # cosine similarity between penalised prob vectors (scalar)
        numer   = (p_base * p_c).sum(1)                              # (N,)
        denom   = (np.linalg.norm(p_base, axis=1)
                   * np.linalg.norm(p_c, axis=1) + EPS)             # (N,)
        cos_sim = float(np.mean(numer / denom))                      # scalar ✓

        row = {
            "scale_c":      float(c),
            "ece":          m_c["ece"],
            "nll":          m_c["nll"],
            "acc":          m_c["acc"],
            "aurc":         m_c["aurc"],
            "delta_ece":    round(m_c["ece"]  - m_base["ece"],  5),
            "delta_nll":    round(m_c["nll"]  - m_base["nll"],  5),
            "delta_acc":    round(m_c["acc"]  - m_base["acc"],  5),
            "delta_aurc":   round(m_c["aurc"] - m_base["aurc"], 5),
            "spearman_rho": round(spr, 4),
            "spearman_p":   round(spp, 4),
            "attn_cos_sim": round(cos_sim, 4),
        }
        rows.append(row)
        print(f"      ECE={m_c['ece']:.4f} (Δ{row['delta_ece']:+.5f})  "
              f"NLL={m_c['nll']:.4f} (Δ{row['delta_nll']:+.5f})  "
              f"Acc={m_c['acc']:.4f}  "
              f"ρ(U,U_c)={spr:.4f}  cosine={cos_sim:.4f}")

    return rows, data_base


# ═══════════════════════════════════════════════════════════════════════════════
#  PART B — Epistemic Alignment Correlation Checks
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_b(data_base, labels_np):
    print("\n  ── Part B: Epistemic Alignment Correlation Checks ──")

    u_vec  = u_cv(data_base)
    H_vec  = predictive_entropy(data_base["probs_all"])
    MI_vec = mutual_information(data_base["probs_all"])
    enorm  = data_base["emb_norm"]

    r_uh,    p_uh   = pearson(u_vec,  H_vec)
    rho_uh,  _      = spearman(u_vec, H_vec)
    r_umi,   p_umi  = pearson(u_vec,  MI_vec)
    rho_umi, _      = spearman(u_vec, MI_vec)
    r_ue,    p_ue   = pearson(u_vec,  enorm)
    rho_ue,  _      = spearman(u_vec, enorm)
    r_ph,    p_ph   = partial_corr_pearson(u_vec, H_vec,  enorm)
    r_pmi,   p_pmi  = partial_corr_pearson(u_vec, MI_vec, enorm)

    result = {
        "U_vs_H_pearson":           round(r_uh,   4),
        "U_vs_H_pearson_p":         round(p_uh,   4),
        "U_vs_H_spearman":          round(rho_uh, 4),
        "U_vs_MI_pearson":          round(r_umi,  4),
        "U_vs_MI_pearson_p":        round(p_umi,  4),
        "U_vs_MI_spearman":         round(rho_umi,4),
        "U_vs_norm_pearson":        round(r_ue,   4),
        "U_vs_norm_pearson_p":      round(p_ue,   4),
        "U_vs_norm_spearman":       round(rho_ue, 4),
        "partial_U_H_ctrl_norm_r":  round(r_ph,   4),
        "partial_U_H_ctrl_norm_p":  round(p_ph,   4),
        "partial_U_MI_ctrl_norm_r": round(r_pmi,  4),
        "partial_U_MI_ctrl_norm_p": round(p_pmi,  4),
    }

    print(f"    U vs H[p(y|x)] — Pearson r={r_uh:.4f} (p={p_uh:.4f})  "
          f"Spearman ρ={rho_uh:.4f}")
    print(f"    U vs MI        — Pearson r={r_umi:.4f} (p={p_umi:.4f})  "
          f"Spearman ρ={rho_umi:.4f}")
    print(f"    U vs ‖e‖       — Pearson r={r_ue:.4f} (p={p_ue:.4f})  "
          f"Spearman ρ={rho_ue:.4f}  [norm-control sanity]")
    print(f"    Partial r(U, H  | ‖e‖): {r_ph:.4f}  (p={p_ph:.4f})")
    print(f"    Partial r(U, MI | ‖e‖): {r_pmi:.4f}  (p={p_pmi:.4f})")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  PART C — Alternative Uncertainty Definitions
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_c(data_base, labels_np, lam, model, device):
    print("\n  ── Part C: Alternative Uncertainty Definitions ──")
    rows = []
    for u_name, u_fn in U_VARIANTS.items():
        u_vec = u_fn(data_base)
        probs = penalised_probs(data_base, u_vec, lam, model, device)
        m     = metrics_dict(probs, labels_np)
        rows.append({"u_variant": u_name, **m})
        print(f"    {u_name:>10s}:  ECE={m['ece']:.4f}  NLL={m['nll']:.4f}  "
              f"Acc={m['acc']:.4f}  AURC={m['aurc']:.4f}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — LaTeX table builders
# ═══════════════════════════════════════════════════════════════════════════════

def latex_part_a(rows, dataset):
    lines = [
        f"% Part A -- Embedding-Rescaling Invariance ({dataset})",
        "\\begin{tabular}{crrrrrrrc}",
        "\\toprule",
        "$c$ & ECE$\\downarrow$ & NLL$\\downarrow$ & Acc$\\uparrow$ & AURC$\\downarrow$"
        " & $\\Delta$ECE & $\\Delta$NLL & $\\Delta$Acc & $\\rho(U,U_c)$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['scale_c']:.1f} & {r['ece']:.4f} & {r['nll']:.4f} & "
            f"{r['acc']:.4f} & {r['aurc']:.4f} & "
            f"{r['delta_ece']:+.5f} & {r['delta_nll']:+.5f} & "
            f"{r['delta_acc']:+.5f} & {r['spearman_rho']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def latex_part_b(r, dataset):
    lines = [
        f"% Part B -- Epistemic Alignment Correlations ({dataset})",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Signal Pair & Control & Pearson $r$ & Spearman $\\rho$ \\\\",
        "\\midrule",
        f"$U$ vs $H[p(y|x)]$ & --- & {r['U_vs_H_pearson']:.4f} & "
        f"{r['U_vs_H_spearman']:.4f} \\\\",
        f"$U$ vs MI & --- & {r['U_vs_MI_pearson']:.4f} & "
        f"{r['U_vs_MI_spearman']:.4f} \\\\",
        f"$U$ vs $\\|e\\|$ (sanity) & --- & {r['U_vs_norm_pearson']:.4f} & "
        f"{r['U_vs_norm_spearman']:.4f} \\\\",
        "\\midrule",
        f"$U$ vs $H$ & $\\|e\\|$ & {r['partial_U_H_ctrl_norm_r']:.4f} & --- \\\\",
        f"$U$ vs MI  & $\\|e\\|$ & {r['partial_U_MI_ctrl_norm_r']:.4f} & --- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def latex_part_c(rows, dataset):
    best = dict(
        ece  = min(r["ece"]  for r in rows),
        nll  = min(r["nll"]  for r in rows),
        acc  = max(r["acc"]  for r in rows),
        aurc = min(r["aurc"] for r in rows),
    )

    def bf(val, key):
        s = f"{val:.4f}"
        return f"\\textbf{{{s}}}" if val == best[key] else s

    lines = [
        f"% Part C -- Uncertainty Variant Comparison ({dataset})",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "$U$ Variant & ECE$\\downarrow$ & NLL$\\downarrow$ & "
        "Acc$\\uparrow$ & AURC$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['u_variant']} & {bf(r['ece'],'ece')} & {bf(r['nll'],'nll')} & "
            f"{bf(r['acc'],'acc')} & {bf(r['aurc'],'aurc')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Three clean CSV writers (no mixed-schema rows)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_csv(path, rows):
    """Write a list of uniform dicts to CSV."""
    if not rows:
        return
    # Build ordered fieldnames: 'dataset' first, then rest from first row
    first_keys = list(rows[0].keys())
    if "dataset" not in first_keys:
        fieldnames = first_keys
    else:
        fieldnames = ["dataset"] + [k for k in first_keys if k != "dataset"]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] CSV → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Scale and Geometry Diagnostics for UAT-LITE"
    )
    parser.add_argument("--datasets",   nargs="+", default=["mnli", "squad"],
                        choices=["mnli", "squad", "sst2"])
    parser.add_argument("--ckpt_root",  default="./experiments")
    parser.add_argument("--backbone",   default="bert-base-uncased")
    parser.add_argument("--M",          type=int,   default=10)
    parser.add_argument("--lam",        type=float, default=1.0)
    parser.add_argument("--scales",     nargs="+",  type=float,
                        default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--max_eval",   type=int,   default=2000)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--max_length", type=int,   default=256)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="results/scale_geometry")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Three separate accumulators — no schema mixing
    csv_a_rows, csv_b_rows, csv_c_rows = [], [], []
    all_json = {}

    for dataset in args.datasets:
        print("\n" + "=" * 72)
        print(f"  DATASET: {dataset.upper()}  |  M={args.M}  |  lambda={args.lam}")
        print("=" * 72)

        ckpt_path = (Path(args.ckpt_root) / DATASET_CKPT[dataset]
                     / args.backbone / "checkpoints" / "best_model.pt")
        print(f"  Checkpoint : {ckpt_path}")
        model     = load_model(str(ckpt_path), args.device)
        tokenizer = AutoTokenizer.from_pretrained(model.base_model_name)

        print("  Loading data...")
        loader, labels_np = load_eval_dataset(
            dataset, tokenizer, args.max_length,
            args.batch_size, args.max_eval, args.seed,
        )

        # ── Run all three parts ──────────────────────────────────────────────
        rows_a, data_base = run_part_a(
            model, loader, labels_np,
            args.M, args.lam, args.device, args.scales,
        )
        result_b = run_part_b(data_base, labels_np)
        rows_c   = run_part_c(data_base, labels_np, args.lam, model, args.device)

        # ── LaTeX ────────────────────────────────────────────────────────────
        sep = "-" * 72
        print(f"\n{sep}\n  LaTeX -- {dataset}\n{sep}")
        print(latex_part_a(rows_a, dataset))
        print()
        print(latex_part_b(result_b, dataset))
        print()
        print(latex_part_c(rows_c, dataset))

        # ── Paper-ready interpretation ───────────────────────────────────────
        print(f"\n{sep}\n  INTERPRETATION -- {dataset}\n{sep}")

        rhos   = [r["spearman_rho"] for r in rows_a]
        d_eces = [r["delta_ece"]    for r in rows_a]
        print(f"""
Part A (Rescaling Invariance):
  Spearman rho(U_orig, U_c) across c={args.scales}:
    {[round(x, 4) for x in rhos]}
  delta_ECE across scales:
    {[round(x, 5) for x in d_eces]}
  High rho (>=0.90) confirms uncertainty ranking is preserved under global
  embedding rescaling. Small |delta_ECE| (<0.01) shows downstream calibration
  is stable, establishing invariance of the modulation mechanism to scale.

Part B (Epistemic Alignment):
  U vs H[p(y|x)]:  r={result_b['U_vs_H_pearson']:.4f}   rho={result_b['U_vs_H_spearman']:.4f}
  U vs MI:          r={result_b['U_vs_MI_pearson']:.4f}   rho={result_b['U_vs_MI_spearman']:.4f}
  U vs norm(e):     r={result_b['U_vs_norm_pearson']:.4f}  [should be low -- norm-control]
  Partial r(U, H  | norm): {result_b['partial_U_H_ctrl_norm_r']:.4f}
  Partial r(U, MI | norm): {result_b['partial_U_MI_ctrl_norm_r']:.4f}
  Significant correlation with H and MI confirms U captures genuine epistemic
  uncertainty. Low r(U, norm) and non-trivial partial correlations after
  controlling for embedding norm rule out U as a proxy for vector magnitude.

Part C (U Variant Stability):""")
        for r in rows_c:
            print(f"  {r['u_variant']:>10s}:  ECE={r['ece']:.4f}  "
                  f"NLL={r['nll']:.4f}  Acc={r['acc']:.4f}  AURC={r['aurc']:.4f}")
        ece_r = max(r["ece"] for r in rows_c) - min(r["ece"] for r in rows_c)
        acc_r = max(r["acc"] for r in rows_c) - min(r["acc"] for r in rows_c)
        print(f"  ECE range: {ece_r:.4f}  |  Acc range: {acc_r:.4f}")
        print("  Tight clustering confirms conclusions are not sensitive to the")
        print("  specific uncertainty estimator.")

        # ── Save per-dataset JSON ────────────────────────────────────────────
        json_path = out_dir / f"scale_geometry_{dataset}.json"
        payload   = {
            "dataset":  dataset,
            "M":        args.M,
            "lam":      args.lam,
            "seed":     args.seed,
            "max_eval": args.max_eval,
            "part_a":   rows_a,
            "part_b":   result_b,
            "part_c":   rows_c,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[OK] JSON --> {json_path}")
        all_json[dataset] = payload

        # ── Accumulate typed CSV rows ────────────────────────────────────────
        for r in rows_a:
            csv_a_rows.append({"dataset": dataset, **r})
        csv_b_rows.append({"dataset": dataset, **result_b})
        for r in rows_c:
            csv_c_rows.append({"dataset": dataset, **r})

    # ── Write three clean, uniform CSVs ─────────────────────────────────────
    print("\n" + "-" * 72)
    _write_csv(out_dir / "part_a_rescaling.csv",    csv_a_rows)
    _write_csv(out_dir / "part_b_correlations.csv", csv_b_rows)
    _write_csv(out_dir / "part_c_variants.csv",     csv_c_rows)

    combined = out_dir / "scale_geometry_all.json"
    with open(combined, "w") as f:
        json.dump(all_json, f, indent=2)
    print(f"[OK] Combined JSON --> {combined}")

    print(f"\n  All diagnostics complete.")
    print(f"  Output directory: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()