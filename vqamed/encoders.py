"""Encoder modules for VQA Medical."""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel


class VisualEncoder(nn.Module):
    """Visual encoder using DenseNet121 backbone."""
    
    def __init__(self, embed_dim: int = 512):
        """
        Initialize Visual Encoder.
        
        Args:
            embed_dim: Output embedding dimension.
        """
        super().__init__()
        base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.features = base_model.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W].
            
        Returns:
            Image embeddings [B, embed_dim].
        """
        x = self.features(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.projection(x)
        return x


class TextEncoder(nn.Module):
    """Text encoder using BERT backbone."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', embed_dim: int = 512):
        """
        Initialize Text Encoder.
        
        Args:
            model_name: Pretrained transformer model name.
            embed_dim: Output embedding dimension.
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.projection = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, T].
            attention_mask: Attention mask [B, T].
            
        Returns:
            Text embeddings [B, embed_dim].
        """
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0]
        return self.projection(cls_token)
