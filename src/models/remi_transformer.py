import torch
import torch.nn as nn
from src.models.positional_encoding import PositionalEncoding

class RemiTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=8,
        n_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, input_ids, src_key_padding_mask=None, **kwargs):
        """
        input_ids: (B, L) int64
        src_key_padding_mask: (B, L) bool, True = padding positions to mask
        """
        seq_len = input_ids.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        x = self.embed(input_ids) * (self.d_model ** 0.5)
        x = self.pos_enc(x)
        x = self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=True)
        logits = self.lm_head(x)  # (B, L, vocab_size)
        return logits
