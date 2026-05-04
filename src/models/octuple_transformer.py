import torch.nn as nn
from src.models.positional_encoding import PositionalEncoding

class OctupleTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_sizes_per_field,  # list[int], e.g. [160, 64, 60, 32, 64, ...]
        d_model=256,
        n_layers=4,
        n_heads=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=4096,
    ):
        super().__init__()
        self.num_fields = len(vocab_sizes_per_field)
        self.d_model = d_model
        self.vocab_sizes = vocab_sizes_per_field

        # One embedding per field
        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(vs, d_model) for vs in vocab_sizes_per_field]
        )

        #self.layer_norm = nn.LayerNorm(d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # One output head per field
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vs) for vs in vocab_sizes_per_field]
        )

    def forward(self, input_ids, attn_mask=None):
        """
        input_ids: (B, L, F) int64, F = num_fields
        """
        B, L, F = input_ids.shape
        assert F == self.num_fields

        # Sum field embeddings → pooled note embedding
        x = 0
        for f in range(self.num_fields):
            x = x + self.field_embeddings[f](input_ids[:, :, f])
        x = x * (1.0 / (self.num_fields ** 0.5))  # scale a bit
        # x = self.layer_norm(x) # Try this if not stable

        x = self.pos_enc(x)
        x = self.encoder(x, mask=attn_mask)

        # Output logits per field: list of (B, L, vocab_f)
        logits_per_field = [head(x) for head in self.heads]
        return logits_per_field
