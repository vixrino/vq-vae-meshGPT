"""
Residual Vector Quantization (RVQ) as used in MeshGPT / SoundStream / Encodec.

Stacks N VQ levels. The first level quantizes the encoder output; each
subsequent level quantizes the residual of the previous level. The final
quantized vector is the sum of all levels:

    z_q = q_1 + q_2 + ... + q_N

where q_i comes from codebook_i applied to (z_e - sum_{j<i} q_j).

Rare / unique triangles can use the higher levels to capture fine detail
while most of the variance is absorbed by the first level.
"""

import torch
import torch.nn as nn

from vector_quantizer import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int = 3,
        num_embeddings: int = 512,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_code_threshold: int = 2,
        reset_every: int = 20,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.levels = nn.ModuleList([
            VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                ema_decay=ema_decay,
                epsilon=epsilon,
                dead_code_threshold=dead_code_threshold,
                reset_every=reset_every,
            )
            for _ in range(num_levels)
        ])

    @torch.no_grad()
    def init_codebook_from_data(self, z_e: torch.Tensor) -> None:
        """Initialize each level's codebook from the residual at that level."""
        residual = z_e
        for level in self.levels:
            level.init_codebook_from_data(residual)
            distances = (
                torch.sum(residual ** 2, dim=1, keepdim=True)
                + torch.sum(level.embedding.weight ** 2, dim=1)
                - 2 * residual @ level.embedding.weight.t()
            )
            indices = torch.argmin(distances, dim=1)
            z_q_i = level.embedding(indices)
            residual = residual - z_q_i

    def forward(
        self, z_e: torch.Tensor, bypass: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if bypass:
            dummy_indices = torch.zeros(
                z_e.shape[0], self.num_levels, dtype=torch.long, device=z_e.device
            )
            losses = {
                "vq_loss": torch.tensor(0.0, device=z_e.device),
                "commitment_loss": torch.tensor(0.0, device=z_e.device),
            }
            return z_e, losses, dummy_indices

        residual = z_e
        z_q_total = torch.zeros_like(z_e)
        all_indices = []
        total_vq_loss = torch.tensor(0.0, device=z_e.device)
        total_commit = torch.tensor(0.0, device=z_e.device)

        for level in self.levels:
            z_q_i, losses_i, indices_i = level(residual, bypass=False)
            z_q_total = z_q_total + z_q_i
            residual = residual - z_q_i
            all_indices.append(indices_i)
            total_vq_loss = total_vq_loss + losses_i["vq_loss"]
            total_commit = total_commit + losses_i["commitment_loss"]

        stacked_indices = torch.stack(all_indices, dim=1)

        losses = {
            "vq_loss": total_vq_loss / self.num_levels,
            "commitment_loss": total_commit / self.num_levels,
        }

        return z_q_total, losses, stacked_indices

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Sum codebook vectors across levels. Supports [N] (level 0 only) or [N, L]."""
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)

        n, num_given = indices.shape
        z_q = torch.zeros(n, self.embedding_dim, device=indices.device)
        for i in range(min(num_given, self.num_levels)):
            z_q = z_q + self.levels[i].embedding(indices[:, i])
        return z_q

    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Mean codebook utilization across levels."""
        if indices.dim() == 1:
            return self.levels[0].get_codebook_usage(indices)

        total = 0.0
        num_levels_in = indices.shape[1]
        for i in range(num_levels_in):
            total += self.levels[i].get_codebook_usage(indices[:, i])
        return total / num_levels_in
