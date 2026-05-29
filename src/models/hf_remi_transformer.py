import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from src.models.positional_encoding import GPTStyleEmbedding

class HFRemiGPT(nn.Module):
    """
    GPT-style decoder-only model (plug-in replacement for your RemiTransformerLM)
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=8,
        n_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=1536,
    ):
        super().__init__()

        self.max_len = max_len

        self.embed = GPTStyleEmbedding(vocab_size, d_model, max_len)

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_len,
            n_ctx=max_len,
            n_embd=d_model,
            n_layer=n_layers,
            n_head=n_heads,
            n_inner=dim_feedforward,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            embd_pdrop=0.0,
            bos_token_id=None,
            eos_token_id=None,
            use_cache=False
        )

        self.transformer = GPT2LMHeadModel(config)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, L)
        attention_mask: (B, L) where 1 = real token, 0 = padding
        """

        inputs_embeds = self.embed(input_ids)

        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )

        return outputs.logits