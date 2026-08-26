# define the global configuration of the small GPT-2 model that we will train

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT_CONFIG_124M:
    # vocabulary size
    vocab_size: int = 50257
    # context length
    context_length: int = 1024
    # embedding dimension
    emb_dim: int = 768
    # number of attention heads
    n_heads: int = 12
    # number of layers
    n_layers: int = 12
    # dropout rate
    drop_rate: float = 0.1
    # qkv bias
    qkv_bias: bool = False
