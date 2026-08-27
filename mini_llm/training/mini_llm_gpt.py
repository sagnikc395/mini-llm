import torch
import torch.nn as nn

from ..config import GPT_CONFIG_124M as cfg


class GPTModel(nn.Module):
    def __init__(self, cfg=cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        # this acts as a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(*[GPTModel(cfg) for _ in range(cfg.n_layers)])
        # this act as a placeholder for LayerNorm
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    # placeholder for a real transformerblock
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    # placeholder for a real layernorm block
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x
