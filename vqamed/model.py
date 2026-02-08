"""Main VQA model combining all components."""

import torch
import torch.nn as nn

from .encoders import VisualEncoder, TextEncoder
from .fusion import CrossAttentionFusion
from .decoder import AnswerDecoder


class VQAModel(nn.Module):
    """Complete VQA model for medical image question answering."""
    
    def __init__(
        self,
        visual_encoder: VisualEncoder,
        text_encoder: TextEncoder,
        fusion: CrossAttentionFusion,
        decoder: AnswerDecoder
    ):
        """
        Initialize VQA Model.
        
        Args:
            visual_encoder: Visual feature extractor.
            text_encoder: Text feature extractor.
            fusion: Multimodal fusion module.
            decoder: Answer generation decoder.
        """
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion
        self.decoder = decoder

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: Input images [B, C, H, W].
            input_ids: Question token IDs [B, T].
            attention_mask: Question attention mask [B, T].
            decoder_input_ids: Decoder input IDs [B, T].
            
        Returns:
            Logits [B, T, vocab_size].
        """
        img_feat = self.visual_encoder(image)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        fused = self.fusion(text_out, img_feat)
        if fused.dim() == 2:
            fused = fused.unsqueeze(1)
            
        logits = self.decoder(
            tgt_input_ids=decoder_input_ids,
            memory=fused
        )

        return logits

    @classmethod
    def from_config(cls, config, vocab_size: int) -> "VQAModel":
        """
        Create model from configuration.
        
        Args:
            config: Configuration object.
            vocab_size: Vocabulary size for decoder.
            
        Returns:
            Initialized VQAModel.
        """
        visual_encoder = VisualEncoder(embed_dim=config.embed_dim)
        text_encoder = TextEncoder(
            model_name=config.text_model_name,
            embed_dim=config.embed_dim
        )
        fusion = CrossAttentionFusion(
            dim=config.embed_dim,
            heads=config.num_heads
        )
        decoder = AnswerDecoder(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            max_len=config.max_len,
            num_layers=config.decoder_layers,
            nhead=config.num_heads
        )
        
        return cls(visual_encoder, text_encoder, fusion, decoder)
