from torch.nn import Module
import torch.nn as nn
import torch


class CausalAttention(Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        # we transpose dimensions 1 and 2, keeping the batch dimension at the first position(0)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            # operations with a trailing underscore are performed in-place, avoiding unnecessary memory copies.
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf,
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print(f"context_vecs.shape: {context_vecs.shape}")
