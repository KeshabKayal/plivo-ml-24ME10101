"""Optimized GPT model.

Key changes from baseline:
  - tie_weights = True  (head.weight == tok_emb.weight, saves vocab*n_embd params)
  - Larger n_embd/n_layer/n_head to use the freed parameter budget
  - block_size = 256 to benefit from BPE's compressed sequences
  - Scaled init on projection layers (GPT-2 style: std / sqrt(2*n_layer))
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 320  # updated by train.py from tokenizer.vocab_size
    block_size = 256  # wider context window (BPE compresses sequence length)
    n_layer = 6  # ─┐
    n_head = 8  #  ├─ calibrated: gives 1,948,160 unique params (vocab=320)
    n_embd = 160  # ─┘  (within 1,900,000–2,000,000 target window)
    dropout = 0.0
    tie_weights = True  # share tok_emb <-> head weights; saves vocab*n_embd params


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        nh = self.n_head
        q = q.view(B, T, nh, C // nh).transpose(1, 2)
        k = k.view(B, T, nh, C // nh).transpose(1, 2)
        v = v.view(B, T, nh, C // nh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init)
        # GPT-2-style scaled init for residual projections
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.2.weight")):
                nn.init.normal_(p, mean=0.0, std=scale)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert (
            T <= self.cfg.block_size
        ), f"Sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def n_params(self):
        """Count unique parameters (respects weight tying — shared counted once)."""
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def n_params_unique(self):
        """Alias for n_params(); kept for compatibility."""
        return self.n_params()
