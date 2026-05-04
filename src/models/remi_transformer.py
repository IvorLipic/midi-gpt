import torch.nn as nn
from src.models.positional_encoding import PositionalEncoding

class RemiTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        dim_feedforward=512,
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
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attn_mask=None):
        """
        input_ids: (B, L) int64
        attn_mask: (L, L) causal mask
        """
        x = self.embed(input_ids) * (self.d_model ** 0.5)
        x = self.pos_enc(x)
        x = self.encoder(x, mask=attn_mask)
        logits = self.lm_head(x)  # (B, L, vocab_size)
        return logits
