"""
Vector Quantization module with EMA codebook updates.

Implements the VQ layer from "Neural Discrete Representation Learning"
(van den Oord et al., 2017) with exponential moving average updates
for codebook stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Maps continuous encoder outputs to discrete codebook entries.

    Uses straight-through estimator for gradient flow:
        forward pass: z_q = codebook lookup (discrete)
        backward pass: gradients pass through to z_e (continuous)

    Codebook is updated via EMA rather than gradient descent for stability.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: Encoder output [N, D]

        Returns:
            z_q: Quantized embeddings [N, D]
            vq_loss: VQ objective (codebook + commitment loss)
            encoding_indices: Index of nearest codebook entry per input [N]
        """
        distances = (
            torch.sum(z_e**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * z_e @ self.embedding.weight.t()
        )

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)

        if self.training:
            self._ema_update(z_e, encoding_indices)

        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())

        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, encoding_indices

    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook via exponential moving average."""
        encodings = F.one_hot(indices, self.num_embeddings).float()

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            encodings.sum(0), alpha=1 - self.ema_decay
        )

        embed_sum = encodings.t() @ z_e
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        # Laplace smoothing to avoid division by zero
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )

        self.embedding.weight.data.copy_(
            self.ema_embed_sum / cluster_size.unsqueeze(1)
        )

    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries used in a batch."""
        unique = torch.unique(indices)
        return len(unique) / self.num_embeddings
