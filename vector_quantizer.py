"""
Vector Quantization module with EMA codebook updates.

Implements the VQ layer from "Neural Discrete Representation Learning"
(van den Oord et al., 2017) with:
- EMA codebook updates
- Dead code reset (replaces unused codes with encoder outputs)
- Diversity loss (encourages uniform codebook usage)
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

    Includes dead code reset and diversity loss to prevent codebook collapse.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 128,
        commitment_cost: float = 1.0,
        diversity_weight: float = 0.1,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_code_threshold: int = 2,
        reset_every: int = 20,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.diversity_weight = diversity_weight
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.dead_code_threshold = dead_code_threshold
        self.reset_every = reset_every

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())
        self.register_buffer("code_usage_count", torch.zeros(num_embeddings))
        self.register_buffer("steps_since_reset", torch.tensor(0))

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            z_e: Encoder output [N, D]

        Returns:
            z_q: Quantized embeddings [N, D]
            losses: Dict with vq_loss, diversity_loss, total_vq_loss
            encoding_indices: Index of nearest codebook entry per input [N]
        """
        z_e_normalized = F.normalize(z_e, dim=-1)
        codebook_normalized = F.normalize(self.embedding.weight, dim=-1)

        distances = (
            torch.sum(z_e_normalized**2, dim=1, keepdim=True)
            + torch.sum(codebook_normalized**2, dim=1)
            - 2 * z_e_normalized @ codebook_normalized.t()
        )

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)

        if self.training:
            self._ema_update(z_e, encoding_indices)
            self.code_usage_count.index_add_(
                0, encoding_indices, torch.ones_like(encoding_indices, dtype=torch.float)
            )
            self.steps_since_reset += 1

            if self.steps_since_reset >= self.reset_every:
                self._reset_dead_codes(z_e)
                self.steps_since_reset.zero_()

        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        diversity_loss = self._diversity_loss(distances)

        total_vq_loss = vq_loss + self.diversity_weight * diversity_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        losses = {
            "vq_loss": vq_loss,
            "diversity_loss": diversity_loss,
            "total_vq_loss": total_vq_loss,
        }

        return z_q, losses, encoding_indices

    def _diversity_loss(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Encourage uniform codebook usage via entropy maximization.

        Converts distances to soft assignments, averages across the batch,
        then computes negative entropy. Minimizing this = more uniform usage.
        """
        soft_assign = F.softmax(-distances, dim=-1)
        avg_probs = soft_assign.mean(dim=0)
        entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        max_entropy = torch.log(torch.tensor(self.num_embeddings, dtype=torch.float, device=distances.device))
        return max_entropy - entropy

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

        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )

        self.embedding.weight.data.copy_(
            self.ema_embed_sum / cluster_size.unsqueeze(1)
        )

    def _reset_dead_codes(self, z_e: torch.Tensor) -> None:
        """Replace dead codebook entries with randomly sampled encoder outputs + noise."""
        dead_mask = self.code_usage_count < self.dead_code_threshold
        num_dead = dead_mask.sum().item()

        if num_dead == 0:
            self.code_usage_count.zero_()
            return

        n = z_e.shape[0]
        if n == 0:
            self.code_usage_count.zero_()
            return

        replace_indices = torch.randint(0, n, (num_dead,), device=z_e.device)
        new_codes = z_e[replace_indices].detach()
        noise = torch.randn_like(new_codes) * 0.02
        new_codes = new_codes + noise

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        self.embedding.weight.data[dead_indices] = new_codes
        self.ema_embed_sum[dead_indices] = new_codes
        self.ema_cluster_size[dead_indices] = self.epsilon * 10

        self.code_usage_count.zero_()

    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries used in a batch."""
        unique = torch.unique(indices)
        return len(unique) / self.num_embeddings
