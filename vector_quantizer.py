"""
Vector Quantization module with EMA codebook updates.

Clean implementation without L2 normalization.
Uses commitment loss only (EMA handles codebook movement).
Supports warmup mode where VQ is bypassed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_code_threshold: int = 2,
        reset_every: int = 20,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.dead_code_threshold = dead_code_threshold
        self.reset_every = reset_every

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())
        self.register_buffer("code_usage_count", torch.zeros(num_embeddings))
        self.register_buffer("steps_since_reset", torch.tensor(0))
        self.register_buffer("initialized", torch.tensor(False))

    def init_codebook_from_data(self, z_e: torch.Tensor) -> None:
        """Initialize codebook from actual encoder outputs (call after warmup)."""
        n = z_e.shape[0]
        if n >= self.num_embeddings:
            indices = torch.randperm(n, device=z_e.device)[:self.num_embeddings]
            self.embedding.weight.data.copy_(z_e[indices].detach())
        else:
            repeats = (self.num_embeddings // n) + 1
            pool = z_e.detach().repeat(repeats, 1)[:self.num_embeddings]
            noise = torch.randn_like(pool) * 0.01
            self.embedding.weight.data.copy_(pool + noise)

        self.ema_embed_sum.copy_(self.embedding.weight.data)
        self.ema_cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def forward(
        self, z_e: torch.Tensor, bypass: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if bypass:
            dummy_indices = torch.zeros(z_e.shape[0], dtype=torch.long, device=z_e.device)
            losses = {
                "vq_loss": torch.tensor(0.0, device=z_e.device),
                "commitment_loss": torch.tensor(0.0, device=z_e.device),
            }
            return z_e, losses, dummy_indices

        # Euclidean distance (no normalization)
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * z_e @ self.embedding.weight.t()
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

        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        losses = {
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
        }

        return z_q_st, losses, encoding_indices

    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        encodings = F.one_hot(indices, self.num_embeddings).float()

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            encodings.sum(0), alpha=1 - self.ema_decay
        )

        embed_sum = encodings.t() @ z_e.detach()
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
        dead_mask = self.code_usage_count < self.dead_code_threshold
        num_dead = dead_mask.sum().item()

        if num_dead > 0 and z_e.shape[0] > 0:
            replace_indices = torch.randint(0, z_e.shape[0], (num_dead,), device=z_e.device)
            new_codes = z_e[replace_indices].detach()
            noise = torch.randn_like(new_codes) * 0.01
            new_codes = new_codes + noise

            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            self.embedding.weight.data[dead_indices] = new_codes
            self.ema_embed_sum[dead_indices] = new_codes
            self.ema_cluster_size[dead_indices] = self.epsilon * 10

        self.code_usage_count.zero_()

    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        unique = torch.unique(indices)
        return len(unique) / self.num_embeddings
