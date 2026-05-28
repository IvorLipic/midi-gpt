import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.positional_encoding import PositionalEncoding


class NestedMultiHeadAttention(nn.Module):
    """
    Multi-head attention supporting nested tensors (torch.jagged) and dense tensors.
    Differences from nn.MultiheadAttention:
    - Batch-first only (nested tensors require this)
    - No add_bias_kv, add_zero_attn, need_weights, average_attn_weights
    - Unnecessary fast-path logic removed
    """
    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias
        self._reset_parameters()

    def _reset_parameters(self):
        # Match nn.MultiheadAttention._reset_parameters:
        # - in_proj_weight: xavier_uniform (default gain=1.0)
        # - in_proj_bias: constant 0.0
        # - out_proj.weight: NOT re-initialized (stays nn.Linear default = kaiming_uniform)
        # - out_proj.bias: constant 0.0
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.packed_proj.weight)
            if self.packed_proj.bias is not None:
                nn.init.zeros_(self.packed_proj.bias)
        else:
            for proj in [self.q_proj, self.k_proj, self.v_proj]:
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Step 1: Input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query = F.linear(query, q_weight, q_bias)
                key = F.linear(key, k_weight, k_bias)
                value = F.linear(value, v_weight, v_bias)
        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2: Split heads
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3: SDPA
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
        )

        # Step 4: Output projection
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.out_proj(attn_output)

        return attn_output


class NestedTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = NestedMultiHeadAttention(
            d_model, d_model, d_model, d_model,
            nhead, dropout=dropout, bias=bias, **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class NestedTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: NestedTransformerEncoderLayer,
        num_layers: int,
        norm=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm
        self.device = device
        self.dtype = dtype

    def forward(self, src, mask=None, is_causal=False):
        output = src
        for mod in self.layers:
            output = mod(output, mask, is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


class NestedRemiTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = NestedTransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = NestedTransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attn_mask=None, is_causal=False):
        """
        Two modes:
        - Training (list): input_ids is a list of (L_i,) 1-D tokens — embed/PE eagerly, create NJT, encoder
        - Generation (dense): input_ids is dense (B, L) tensor
        """
        if isinstance(input_ids, list):
            # Training path: embed + PE per-item (eager), then NJT encoder
            pe = self.pos_enc.pe.squeeze(0)
            x_list = [self.embed(t) * (self.d_model ** 0.5) for t in input_ids]
            x_list = [xi + pe[:xi.size(0)] for xi in x_list]
            x = torch.nested.as_nested_tensor(x_list, layout=torch.jagged)
            x = self.encoder(x, is_causal=True)
        else:
            # Generation path: dense
            x = self.embed(input_ids) * (self.d_model ** 0.5)
            x = self.pos_enc(x)
            x = self.encoder(x, mask=attn_mask, is_causal=is_causal)

        logits = self.lm_head(x)
        return logits
