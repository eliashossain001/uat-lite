"""Uncertainty-weighted self-attention (paper Sec. "Uncertainty-Weighted Self-Attention").

Pre-softmax scores a_ij = Q_i K_j^T / sqrt(d_k) are attenuated as

    a~_ij = a_ij * f(u_ij),   f(u) = exp(-lambda * u)   (default, "exponential")

with u_ij = U(x_i) (Q-only), U(x_j) (K-only) or U(x_i) + U(x_j) (QK). The V and
QKV variants additionally scale the values: V~_j = V_j * f(U(x_j)).

``UncertaintyAttention`` temporarily replaces the forward of every BERT-style
self-attention module (one with query/key/value/transpose_for_scores) inside a
``with`` block, so the model behaves as a plain encoder outside of it.
"""

import math
import types

import torch
import torch.nn.functional as F

VARIANTS = ("q", "k", "qk", "v", "qkv", "none")

PENALTIES = {
    "exponential": lambda u, lam: torch.exp(-lam * u),
    "linear": lambda u, lam: torch.clamp(1.0 - lam * u, min=0.0),
    "reciprocal": lambda u, lam: 1.0 / (1.0 + lam * u),
}


def _is_self_attention(m):
    return all(hasattr(m, a) for a in ("query", "key", "value", "transpose_for_scores"))


class UncertaintyAttention:
    """Context manager that patches self-attention with uncertainty modulation.

    Call ``set_uncertainty(U)`` before each encoder pass; U is [B, S] (or None
    for unmodulated attention).
    """

    def __init__(self, transformer, lam=0.5, variant="q", penalty="exponential", attn_dropout=0.0):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.modules = [m for m in transformer.modules() if _is_self_attention(m)]
        if not self.modules:
            raise RuntimeError("no BERT-style self-attention modules found to patch")
        self.lam, self.variant, self.attn_dropout = float(lam), variant, float(attn_dropout)
        self.f = PENALTIES[penalty]
        self.U = None
        self._orig = []

    def set_uncertainty(self, U):
        self.U = U

    def _score_factor(self):
        U = self.U
        if self.variant in ("q",):
            return self.f(U, self.lam)[:, None, :, None]
        if self.variant in ("k",):
            return self.f(U, self.lam)[:, None, None, :]
        if self.variant in ("qk", "qkv"):
            return self.f(U[:, :, None] + U[:, None, :], self.lam)[:, None]
        return None

    def __enter__(self):
        ctx = self

        def forward(module, hidden_states, attention_mask=None, head_mask=None,
                    encoder_hidden_states=None, encoder_attention_mask=None,
                    past_key_value=None, output_attentions=False):
            if encoder_hidden_states is not None or past_key_value is not None:
                raise NotImplementedError("UncertaintyAttention supports encoder self-attention only")
            q = module.transpose_for_scores(module.query(hidden_states))
            k = module.transpose_for_scores(module.key(hidden_states))
            v = module.transpose_for_scores(module.value(hidden_states))
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
            if ctx.U is not None and ctx.variant != "none":
                factor = ctx._score_factor()
                if factor is not None:
                    scores = scores * factor
                if ctx.variant in ("v", "qkv"):
                    v = v * ctx.f(ctx.U, ctx.lam)[:, None, :, None]
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = torch.softmax(scores, dim=-1)
            if ctx.attn_dropout > 0:
                probs = F.dropout(probs, p=ctx.attn_dropout, training=True)
            if head_mask is not None:
                probs = probs * head_mask
            context = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
            context = context.view(*context.shape[:-2], -1)
            return (context, probs) if output_attentions else (context,)

        for m in self.modules:
            self._orig.append(m.forward)
            m.forward = types.MethodType(forward, m)
        return self

    def __exit__(self, *exc):
        for m, f in zip(self.modules, self._orig):
            m.forward = f
        self._orig = []
        self.U = None
        return False


class RunningTokenUncertainty:
    """Welford running estimate of U(x_j) = mean_d Std_m(z_{j,d}^(m)) over embedding samples."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def value(self, like):
        """Current estimate [B, S]; zeros until two samples have been seen (U^(0) = 0)."""
        if self.n < 2:
            return torch.zeros(like.shape[:2], device=like.device, dtype=like.dtype)
        return (self.m2 / (self.n - 1)).clamp(min=0).sqrt().mean(dim=-1)

    def update(self, z):
        z = z.detach().float()
        self.n += 1
        if self.mean is None:
            self.mean, self.m2 = z.clone(), torch.zeros_like(z)
            return
        delta = z - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (z - self.mean)
