import torch
from torch.nn import Module, ModuleList, Linear, Dropout
from compact_causal_attention import CausalAttention


class MultiHeadAttentionWrapper(Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


## an efficient implementation of multiheaded attention
class MultiHeadAttention(Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        # reduce the projection dim to match the desired output dim
        self.head_dum = d_out // num_heads
        self.w_query = Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = Linear(d_in, d_out, bias=qkv_bias)

        # use a linear layer to combine the head output
        self.out_proj = Linear(d_out, d_out)
        self.dropout = Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.w_key(x)
        # tensor shape would be (b,num_tokens,d_out)
        queries = self.w_query(x)
        values = self.w_value(x)
        # here we implicitly split the matrix by adding a num_heads dimension.
        # then we unrol the last dim (b,num_tokens,d_out) -> (b,num_tokens,num_heads,head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # then transpose from the shape (b,num_tokens, num_heads, head_dum) to (b,num_heads,num_tokens,head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.tranpose(1, 2)
        values = values.transpose(1, 2)

        # then compute the dot product for each of the head
        attn_scores = queries @ keys.transpose(2, 3)
        # mask the truncated to the number of tokens
        mask_bool = self.mask_bool()[:num_tokens, :num_tokens]
        # now use the mask to fill the attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = self.dropout(attn_weights)
        # tensor shape is (b,num_tokens,n_heads,head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # combine the heads, where self.d_out = self.num_heads @ self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # add a linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec
