"""Answer decoder module for VQA Medical."""

import torch
import torch.nn as nn


class AnswerDecoder(nn.Module):
    """Transformer decoder for generating answers."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        max_len: int = 32,
        num_layers: int = 4,
        nhead: int = 8
    ):
        """
        Initialize Answer Decoder.
        
        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Embedding dimension.
            max_len: Maximum sequence length.
            num_layers: Number of transformer decoder layers.
            nhead: Number of attention heads.
        """
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.transformer = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len
        self.embed_dim = embed_dim

    def forward(
        self,
        tgt_input_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tgt_input_ids: Target token IDs [B, T].
            memory: Encoder output [B, S, D].
            tgt_key_padding_mask: Optional padding mask.
            
        Returns:
            Logits [B, T, vocab_size].
        """
        B, T = tgt_input_ids.size()
        device = tgt_input_ids.device

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        tgt_emb = self.token_emb(tgt_input_ids) + self.pos_emb(pos_ids)

        causal_mask = torch.triu(
            torch.ones((T, T), device=device),
            diagonal=1
        ).bool()

        out = self.transformer(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask
        )

        return self.fc_out(out)
