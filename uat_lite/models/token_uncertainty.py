"""Stochastic embeddings and token-level uncertainty U(x_j).

MC dropout is applied to the encoder's embedding output. The token-level
uncertainty of token j is the standard deviation of its embedding across
dropout samples, averaged over the hidden dimension.
"""

import torch
import torch.nn as nn


class StochasticEmbedding(nn.Module):
    """Dropout on the embedding output, kept active at inference time."""

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

    def enable(self):
        """Keep dropout stochastic even when the parent model is in eval mode."""
        self.dropout.train()

    def sample(self, embeddings: torch.Tensor) -> torch.Tensor:
        """One stochastic embedding sample, [B, S, H]."""
        return self.dropout(embeddings)

    def token_uncertainty(self, embeddings: torch.Tensor, num_samples: int) -> torch.Tensor:
        """U(x_j): std over ``num_samples`` dropout samples, mean over hidden dim -> [B, S]."""
        samples = torch.stack([self.sample(embeddings) for _ in range(num_samples)], dim=0)
        return samples.std(dim=0).mean(dim=-1)
