"""Checkpoint loading and the stochastic-inference estimators used in the paper.

Every estimator loops over a DataLoader whose batches contain ``input_ids``,
``attention_mask`` and a label tensor under ``label_key``, and returns CPU
tensors. All of them run under ``torch.no_grad()``.

Estimators
----------
predict_deterministic       single pass, all dropout off                       -> logits
predict_mc_dropout          all dropout on, M passes, mean of softmax          -> probs
predict_mc_dropout_logits   all dropout on, M passes, softmax of mean logits   -> logits, probs
predict_uat                 UATLite.forward_uat (paper Algorithm 1): uncertainty-
                            weighted attention with the lagged online token
                            uncertainty and component-wise dropout              -> logits, probs
predict_embedding_mc        UATLite.forward_with_mc_sampling: dropout on the
                            embedding output only, deterministic encoder and
                            head, no attention modulation, softmax of mean logits -> logits, probs
predict_pooling_proxy       all dropout on; token uncertainty = std of the
                            last hidden state over M passes; the [CLS] /
                            pooled representation is reweighted by
                            exp(-lambda * U) according to ``variant``           -> logits

Which scripts report which estimator as "UAT-LITE" is listed in the README.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .models import UATLite


def resolve_checkpoint(ckpt_path):
    """Local path, or ``hf://<user>/<repo>/<file>`` downloaded from the Hugging Face Hub."""
    if str(ckpt_path).startswith("hf://"):
        from huggingface_hub import hf_hub_download
        user, repo, filename = str(ckpt_path)[5:].split("/", 2)
        return hf_hub_download(repo_id=f"{user}/{repo}", filename=filename)
    return ckpt_path


def load_checkpoint(ckpt_path, device="cpu"):
    """Load a checkpoint written by ``scripts/train/train.py`` (dict with 'config').

    ``ckpt_path`` may be a local file or ``hf://<user>/<repo>/<file>``.
    """
    ckpt = torch.load(resolve_checkpoint(ckpt_path), map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "config" in ckpt):
        raise RuntimeError(f"Unrecognised checkpoint format: {ckpt_path}")
    model = UATLite(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def enable_dropout(model):
    """Put every nn.Dropout module in train mode; everything else stays in eval."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _cls_logits(model, ids, mask, head_dropout):
    out = model.transformer(input_ids=ids, attention_mask=mask)
    pooled = out.last_hidden_state[:, 0]
    if head_dropout:
        pooled = model.dropout(pooled)
    return model.classifier(pooled)


@torch.no_grad()
def predict_deterministic(model, loader, device, label_key="labels", desc="Base Det."):
    """Returns (logits [N,C], labels [N])."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        all_logits.append(_cls_logits(model, ids, mask, head_dropout=False).cpu())
        all_labels.append(batch[label_key])
    return torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def predict_mc_dropout(model, loader, M, device, label_key="labels"):
    """Global MC dropout (Gal & Ghahramani, 2016). Returns (mean probs, labels)."""
    model.eval()
    enable_dropout(model)
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc=f"MC Global M={M}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        prob_acc = torch.zeros(ids.size(0), model.num_classes, device=device)
        for _ in range(M):
            prob_acc += F.softmax(_cls_logits(model, ids, mask, head_dropout=True), dim=-1)
        all_probs.append((prob_acc / M).cpu())
        all_labels.append(batch[label_key])
    return torch.cat(all_probs), torch.cat(all_labels)


@torch.no_grad()
def predict_mc_dropout_logits(model, loader, M, device, label_key="labels"):
    """Global MC dropout, averaging logits. Returns (mean logits, softmax(mean logits), labels)."""
    model.eval()
    enable_dropout(model)
    all_logits, all_probs, all_labels = [], [], []
    for batch in tqdm(loader, desc=f"MC logits M={M}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logit_acc = torch.zeros(ids.size(0), model.num_classes, device=device)
        for _ in range(M):
            logit_acc += _cls_logits(model, ids, mask, head_dropout=True)
        mean_logits = logit_acc / M
        all_logits.append(mean_logits.cpu())
        all_probs.append(F.softmax(mean_logits, dim=-1).cpu())
        all_labels.append(batch[label_key])
    return torch.cat(all_logits), torch.cat(all_probs), torch.cat(all_labels)


@torch.no_grad()
def predict_embedding_mc(model, loader, M, device, label_key="labels"):
    """MC over embedding dropout via ``UATLite.forward_with_mc_sampling``.

    Returns (mean logits, softmax(mean logits), labels).
    """
    model.eval()
    all_logits, all_probs, all_labels = [], [], []
    for batch in tqdm(loader, desc=f"UAT-LITE M={M}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        mc_out = model.forward_with_mc_sampling(input_ids=ids, attention_mask=mask, num_samples=M)
        mean_logits = mc_out["logits_samples"].mean(dim=0)
        all_logits.append(mean_logits.cpu())
        all_probs.append(F.softmax(mean_logits, dim=-1).cpu())
        all_labels.append(batch[label_key])
    return torch.cat(all_logits), torch.cat(all_probs), torch.cat(all_labels)


@torch.no_grad()
def predict_uat(model, loader, M, device, label_key="labels", lam=None, variant="q",
                penalty="exponential", dropout_rates=None):
    """UAT-LITE (paper Algorithm 1). Returns (mean logits, softmax(mean logits), labels)."""
    kwargs = {"lam": lam, "variant": variant, "penalty": penalty}
    if dropout_rates is not None:
        kwargs["dropout_rates"] = dropout_rates
    model.eval()
    all_logits, all_probs, all_labels = [], [], []
    for batch in tqdm(loader, desc=f"UAT-LITE[{variant}] M={M}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        mean_logits = model.forward_uat(ids, mask, num_samples=M, **kwargs)["logits_samples"].mean(dim=0)
        all_logits.append(mean_logits.cpu())
        all_probs.append(F.softmax(mean_logits, dim=-1).cpu())
        all_labels.append(batch[label_key])
    return torch.cat(all_logits), torch.cat(all_probs), torch.cat(all_labels)


# Default estimator reported as "UAT-LITE" by every evaluation script.
DEFAULT_UAT = "attention"

# Estimators selectable as "UAT-LITE" via ``--uat`` in the evaluation scripts.
# Each returns (mean logits, probs, labels).
UAT_ESTIMATORS = {
    "attention": lambda model, loader, M, device: predict_uat(model, loader, M, device),
    "embedding": lambda model, loader, M, device: predict_embedding_mc(model, loader, M, device),
    "mc_logits": lambda model, loader, M, device: predict_mc_dropout_logits(model, loader, M, device),
}
UAT_HELP = ("UAT-LITE estimator: 'attention' = paper Algorithm 1 (uncertainty-weighted attention); "
            "'embedding' = embedding-only MC dropout (estimator behind the submitted tables); "
            "'mc_logits' = global MC dropout with logit averaging (submitted selective-prediction table)")


def token_uncertainty(hidden_stack):
    """[M, B, S, H] hidden states -> [B, S]: std over passes, mean over hidden dim."""
    return hidden_stack.std(dim=0).mean(dim=-1)


def pool_with_uncertainty(h, token_unc, attention_mask, lam, variant):
    """Reweight one pass's last hidden state ``h`` [B,S,H] into a pooled [B,H] vector.

    variant "q":   [CLS] scaled by exp(-lam * U_cls)
            "k":   mask-normalized weighted mean over tokens, weights exp(-lam * U_j)
            "v":   [CLS] of h * exp(-lam * U)   (equals "q" at the [CLS] position)
            "qkv": "v" gating, then "k" pooling, then "q" scaling
    """
    if variant == "q":
        q_scale = torch.exp(-lam * token_unc[:, 0])
        return h[:, 0] * q_scale.unsqueeze(-1)
    if variant == "k":
        w = torch.exp(-lam * token_unc) * attention_mask.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        return (h * w.unsqueeze(-1)).sum(dim=1)
    if variant == "v":
        v_gate = torch.exp(-lam * token_unc)
        return (h * v_gate.unsqueeze(-1))[:, 0]
    if variant == "qkv":
        v_gate = torch.exp(-lam * token_unc)
        gated = h * v_gate.unsqueeze(-1)
        w = torch.exp(-lam * token_unc) * attention_mask.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        pooled = (gated * w.unsqueeze(-1)).sum(dim=1)
        q_scale = torch.exp(-lam * token_unc[:, 0])
        return pooled * q_scale.unsqueeze(-1)
    raise ValueError(f"Unknown variant: {variant}")


@torch.no_grad()
def predict_pooling_proxy(model, loader, M, lam, device, variant="q", label_key="labels"):
    """Uncertainty-reweighted pooling on top of global MC dropout.

    1. M stochastic passes through the encoder (all dropout on).
    2. U = std of the last hidden state over the M passes, mean over hidden dim.
    3. For each pass, pool with ``pool_with_uncertainty``, apply head dropout
       and the classifier; average the M logits.

    The encoder's attention is NOT modified. Returns (mean logits, labels).
    """
    model.eval()
    enable_dropout(model)
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc=f"MC {variant} lam={lam} M={M}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        hidden = [model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state
                  for _ in range(M)]
        unc = token_uncertainty(torch.stack(hidden, dim=0))
        logit_acc = torch.zeros(ids.size(0), model.num_classes, device=device)
        for h in hidden:
            pooled = pool_with_uncertainty(h, unc, mask, lam, variant)
            logit_acc += model.classifier(model.dropout(pooled))
        all_logits.append((logit_acc / M).cpu())
        all_labels.append(batch[label_key])
    return torch.cat(all_logits), torch.cat(all_labels)
