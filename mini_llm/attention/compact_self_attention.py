import torch.nn as nn 
import torch 
from basic_attention import inputs

class SelfAttention(nn.Module):
    def __init__(self,d_in,d_out):
        # initializes trainable weight matrices for queries, keys and values, each transforming the input dimensions d_in to an output dimension d_out 
        super().__init__()
        self.w_query = nn.Parameter(torch.rand(d_in,d_out))
        self.w_key = nn.Parameter(torch.rand(d_in,d_out))
        self.w_value = nn.Parameter(torch.rand(d_in,d_out))
        
    def forward(self,x):
        # using the forward method,we compute the attention scores (attn_scores) by multiplying queries and keys, normalizing these scores using softmax.
        keys = x @ self.w_key
        queries = x @ self.w_query
        values = x @ self.w_value
        
        attn_scores = queries @ keys.T # omega 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5 , dim=-1
        )
        context_vec = attn_weights @ values 
        return context_vec
    
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in,d_out,qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value = nn.Linear(d_in,d_out,bias=qkv_bias)
    
    def forward(self,x):
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)
        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values 
        return context_vec
    
d_in = inputs.shape[1]
# output embedding size, d_out = 2
d_out = 2

torch.manual_seed(789)
sav2 = SelfAttentionV2(d_in,d_out)
print(sav2(inputs))

## adding causal mask 
# reuse the query and key weight matrices of the SelfAttention object from the previous section for convenience
queries = sav2.w_query(inputs)
keys = sav2.w_key(inputs)
attn_scores = queries @ keys.T 
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5 ,dim=-1)
print(attn_weights)

# tril function to create a mask where the values above the diagonal are 0 

contenxt_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(contenxt_length,contenxt_length))
print(mask_simple)