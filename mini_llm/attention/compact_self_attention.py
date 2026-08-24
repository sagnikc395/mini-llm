import torch.nn as nn 
import torch 

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