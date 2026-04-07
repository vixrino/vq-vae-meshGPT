"""
Graph Neural Network encoder using SAGEConv layers.

Takes a triangle graph (node features = 9D vertex coords) and produces
a latent embedding per triangle suitable for vector quantization.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphEncoder(nn.Module):
    """
    4-layer GraphSAGE encoder.

    Architecture:
        SAGEConv(9 → 64)   + ReLU + LayerNorm
        SAGEConv(64 → 128)  + ReLU + LayerNorm
        SAGEConv(128 → 256) + ReLU + LayerNorm
        SAGEConv(256 → 256) + LayerNorm  (no activation before VQ)

    The final 256-dim embedding per triangle is sent to the vector quantizer.
    """

    def __init__(self, in_channels: int = 9, latent_dim: int = 256):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, 64)
        self.conv2 = SAGEConv(64, 128)
        self.conv3 = SAGEConv(128, 256)
        self.conv4 = SAGEConv(256, latent_dim)

        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(latent_dim)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, 9]
            edge_index: Graph connectivity [2, E]

        Returns:
            z_e: Pre-quantization embeddings [N, latent_dim]
        """
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.conv4(x, edge_index)
        x = self.norm4(x)

        return x
