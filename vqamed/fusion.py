"""Fusion module for VQA Medical."""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion layer for multimodal features."""
    
    def __init__(self, dim: int = 512, heads: int = 8):
        """
        Initialize Cross-Attention Fusion.
        
        Args:
            dim: Feature dimension.
            heads: Number of attention heads.
        """
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention.
        
        Args:
            text_embeds: Text embeddings [B, T, D].
            image_embeds: Image embeddings [B, N, D].
            
        Returns:
            Fused embeddings [B, T, D].
        """
        attn_output, _ = self.cross_attn(
            query=text_embeds,
            key=image_embeds,
            value=image_embeds
        )
        x = self.norm(attn_output + text_embeds)
        x = self.mlp(x)
        return x
