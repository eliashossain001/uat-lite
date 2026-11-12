"""UAT-LITE model.

A pretrained encoder with three inference-time components:
  * ``StochasticEmbedding``  - MC dropout on the embedding output; token-level
                               uncertainty U(x_j) (``token_uncertainty.py``)
  * ``ConfidenceDecision``   - entropy-based confidence and abstention
                               (``confidence_decision.py``)
  * ``LayerWiseVarianceDecomposition`` - layer-wise uncertainty attribution
                               (``variance_decomposition.py``)

Inference paths (both average logits over M passes):
  ``forward_uat``               paper Algorithm 1: uncertainty-weighted attention
                                with the lagged online estimate U^(m-1) and
                                component-wise dropout (0.1 / 0.2 / 0.3).
  ``forward_with_mc_sampling``  dropout on the embedding output only, deterministic
                                encoder, no attention modulation.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .attention import RunningTokenUncertainty, UncertaintyAttention
from .confidence_decision import ConfidenceDecision
from .token_uncertainty import StochasticEmbedding
from .variance_decomposition import LayerWiseVarianceDecomposition

# Number of embedding-dropout samples used to estimate U(x_j) in forward_with_mc_sampling.
TOKEN_UNCERTAINTY_SAMPLES = 3
# Inference-time dropout (embeddings, attention, feed-forward) for forward_uat.
DEFAULT_DROPOUT_RATES = (0.1, 0.2, 0.3)


class UATLite(nn.Module):

    def __init__(
        self,
        base_model_name: str,
        num_classes: int,
        mc_samples: int = 10,
        dropout_rate: float = 0.3,
        uncertainty_penalty: float = 0.5,
        confidence_threshold: float = 0.7,
        use_layer_decomposition: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        self.uncertainty_penalty = uncertainty_penalty  # lambda
        self.confidence_threshold = confidence_threshold  # tau
        self.use_layer_decomposition = use_layer_decomposition

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.transformer = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads

        self.stochastic_embedding = StochasticEmbedding(dropout_rate)
        self.confidence_decision = ConfidenceDecision(
            num_classes=num_classes, confidence_threshold=confidence_threshold, temperature=1.0)
        if use_layer_decomposition:
            self.variance_decomposition = LayerWiseVarianceDecomposition(
                num_layers=self.num_layers, hidden_size=self.hidden_size)

        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    # ------------------------------------------------------------------
    # MC inference
    # ------------------------------------------------------------------
    def forward_with_mc_sampling(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        collect_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """M stochastic passes with dropout on the embedding output only (no attention
        modulation; this is the estimator behind the paper's reported tables).

        Returns:
            logits_samples          [M, B, C]
            token_uncertainty       [B, S]  U(x_j) from min(3, M) embedding samples
            all_layer_hidden_states list of L tensors [M, B, S, H] if ``collect_layers``,
                                    otherwise [1, B, S, H] from the last pass
        """
        if num_samples is None:
            num_samples = self.mc_samples
        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            base_embeddings = self.transformer.embeddings(input_ids)

        self.stochastic_embedding.enable()
        token_uncertainty = self.stochastic_embedding.token_uncertainty(
            base_embeddings, min(TOKEN_UNCERTAINTY_SAMPLES, num_samples))

        if attention_mask is not None:
            # [B, S] -> additive mask [B, 1, 1, S]
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.float32)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        logits_samples, layer_hidden, per_pass = [], None, []
        for m in range(num_samples):
            last = m == num_samples - 1
            encoder_out = self.transformer.encoder(
                self.stochastic_embedding.sample(base_embeddings),
                attention_mask=extended_mask,
                output_hidden_states=last or collect_layers,
            )
            if collect_layers:
                per_pass.append(encoder_out.hidden_states[1:])
            elif last and encoder_out.hidden_states is not None:
                layer_hidden = [h.unsqueeze(0) for h in encoder_out.hidden_states[1:]]
            pooled = self.dropout(encoder_out.last_hidden_state[:, 0])
            logits_samples.append(self.classifier(pooled))
            del encoder_out, pooled
            if not last:
                torch.cuda.empty_cache()

        if collect_layers:
            layer_hidden = [torch.stack([p[l] for p in per_pass], dim=0) for l in range(self.num_layers)]
        if layer_hidden is None:
            layer_hidden = [torch.zeros(1, batch_size, seq_len, self.hidden_size, device=input_ids.device)
                            for _ in range(self.num_layers)]
        return {
            "logits_samples": torch.stack(logits_samples, dim=0),
            "all_layer_hidden_states": layer_hidden,
            "token_uncertainty": token_uncertainty,
        }

    @torch.no_grad()
    def forward_uat(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        lam: Optional[float] = None,
        variant: str = "q",
        penalty: str = "exponential",
        dropout_rates: Tuple[float, float, float] = DEFAULT_DROPOUT_RATES,
        collect_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """UAT-LITE inference (paper Algorithm 1), single-pass online protocol.

        For m = 1..M: sample an embedding z^(m) (dropout ``p_emb``), run the
        encoder with attention modulated by the lagged estimate U^(m-1)
        (U^(0) = 0) and with dropout ``p_attn`` on attention probabilities and
        the attention output and ``p_ffn`` on the feed-forward output, compute
        logits l^(m) from [CLS], then update U^(m) from z^(m) (Welford).

        Args:
            lam: uncertainty penalty lambda (default: ``self.uncertainty_penalty``)
            variant: "q" (default), "k", "qk", "v", "qkv", or "none" (no modulation)
            penalty: "exponential", "linear" or "reciprocal"
            dropout_rates: (p_emb, p_attn, p_ffn); paper default (0.1, 0.2, 0.3)
            collect_layers: also return per-layer hidden states of every pass,
                as needed by the layer-wise variance decomposition
        Returns:
            logits_samples [M, B, C], token_uncertainty U^(M) [B, S],
            and all_layer_hidden_states (list of L tensors [M, B, S, H]) if requested.
        """
        M = num_samples or self.mc_samples
        lam = self.uncertainty_penalty if lam is None else lam
        p_emb, p_attn, p_ffn = dropout_rates
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        ext_mask = self.transformer.get_extended_attention_mask(attention_mask, input_ids.shape)

        was_training = self.training
        self.eval()
        base = self.transformer.embeddings(input_ids)
        hidden_dropouts = {name: m for name, m in self.transformer.encoder.named_modules()
                           if isinstance(m, nn.Dropout)}
        saved_p = {name: m.p for name, m in hidden_dropouts.items()}
        for name, m in hidden_dropouts.items():
            m.p = p_attn if name.endswith("attention.output.dropout") else p_ffn
            m.train()

        running = RunningTokenUncertainty()
        logits, layers = [], []
        try:
            with UncertaintyAttention(self.transformer, lam, variant, penalty, p_attn) as uat:
                for _ in range(M):
                    z = nn.functional.dropout(base, p=p_emb, training=True)
                    uat.set_uncertainty(running.value(z))
                    out = self.transformer.encoder(z, attention_mask=ext_mask,
                                                   output_hidden_states=collect_layers)
                    logits.append(self.classifier(out.last_hidden_state[:, 0]))
                    if collect_layers:
                        layers.append(out.hidden_states[1:])
                    running.update(z)
        finally:
            for name, m in hidden_dropouts.items():
                m.p = saved_p[name]
                m.eval()
            self.train(was_training)

        result = {"logits_samples": torch.stack(logits, dim=0),
                  "token_uncertainty": running.value(base)}
        if collect_layers:
            result["all_layer_hidden_states"] = [torch.stack([p[l] for p in layers], dim=0)
                                                 for l in range(self.num_layers)]
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True,
        return_layer_decomposition: bool = False,
        method: str = "embedding",
        **uat_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training / ``return_uncertainty=False``: one standard pass.

        Eval: M-sample MC inference. ``method="embedding"`` uses
        ``forward_with_mc_sampling``; ``method="attention"`` uses ``forward_uat``
        (``uat_kwargs`` are passed through, e.g. variant, lam, penalty)."""
        batch_size = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self.training or not return_uncertainty:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.classifier(self.dropout(outputs.last_hidden_state[:, 0]))
            decision = self.confidence_decision(logits=logits, return_all=True, apply_temperature=True)
            zeros = torch.zeros(batch_size, device=input_ids.device)
            result = {
                "logits": logits,
                "probabilities": decision["probabilities"],
                "predictions": decision["predictions"],
                "confidence": decision["confidence"],
                "epistemic_uncertainty": zeros,
                "aleatoric_uncertainty": zeros.clone(),
                "should_abstain": decision["should_abstain"],
                "token_uncertainty": torch.zeros(batch_size, input_ids.shape[1], device=input_ids.device),
            }
            if labels is not None:
                result["loss"] = nn.CrossEntropyLoss()(logits, labels)
            return result

        if method == "attention":
            mc = self.forward_uat(input_ids, attention_mask, collect_layers=return_layer_decomposition, **uat_kwargs)
        elif method == "embedding":
            mc = self.forward_with_mc_sampling(input_ids, attention_mask, collect_layers=return_layer_decomposition)
        else:
            raise ValueError(f"unknown method: {method}")
        logits_samples = mc["logits_samples"]
        mean_logits = logits_samples.mean(dim=0)
        decision = self.confidence_decision(logits=mean_logits, return_all=True, apply_temperature=True)

        probs_samples = torch.softmax(logits_samples, dim=-1)
        entropy_samples = -(probs_samples * torch.log(probs_samples + 1e-10)).sum(dim=-1)
        result = {
            "logits": mean_logits,
            "probabilities": decision["probabilities"],
            "predictions": decision["predictions"],
            "confidence": decision["confidence"],
            "epistemic_uncertainty": logits_samples.var(dim=0).mean(dim=-1),
            "aleatoric_uncertainty": entropy_samples.mean(dim=0),
            "should_abstain": decision["should_abstain"],
            "token_uncertainty": mc["token_uncertainty"],
        }

        if return_layer_decomposition and self.use_layer_decomposition:
            try:
                d = self.variance_decomposition(
                    all_layer_hidden_states=mc["all_layer_hidden_states"],
                    logits_samples=logits_samples,
                    return_critical_layers=True,
                )
                result.update({
                    "layer_uncertainties": d["layer_total"],
                    "layer_aleatoric": d["layer_aleatoric"],
                    "layer_epistemic": d["layer_epistemic"],
                    "critical_layers": d["critical_layer_indices"],
                })
            except Exception as e:
                print(f"Warning: layer-wise decomposition failed: {e}")

        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(mean_logits, labels)
        return result

    def predict_with_abstention(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predictions with abstained examples set to -1."""
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_uncertainty=True)
        preds = out["predictions"].clone()
        preds[out["should_abstain"]] = -1
        return preds, out["confidence"], out["should_abstain"]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def get_config(self) -> Dict:
        return {
            "base_model_name": self.base_model_name,
            "num_classes": self.num_classes,
            "mc_samples": self.mc_samples,
            "dropout_rate": self.dropout_rate,
            "uncertainty_penalty": self.uncertainty_penalty,
            "confidence_threshold": self.confidence_threshold,
            "use_layer_decomposition": self.use_layer_decomposition,
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        """Also accepts checkpoints from the earlier code version, which stored a
        duplicate reference to ``transformer.embeddings`` under ``bayesian_embedding.embedding_layer``."""
        legacy = "bayesian_embedding.embedding_layer."
        if any(k.startswith(legacy) for k in state_dict):
            state_dict = dict(state_dict)
            for k in [k for k in state_dict if k.startswith(legacy)]:
                v = state_dict.pop(k)
                ref = state_dict.get("transformer.embeddings." + k[len(legacy):])
                if ref is not None and not torch.equal(ref, v):
                    raise ValueError(f"Legacy key {k} differs from transformer.embeddings")
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device)

    def save_pretrained(self, save_path: str):
        torch.save({"config": self.get_config(), "model_state_dict": self.state_dict()}, save_path)
