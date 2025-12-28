"""
nanozero/model.py - Transformer policy-value network
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanozero.config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AlphaZeroTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(3, config.n_embd)
        self.pos_emb = nn.Embedding(config.board_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.policy_head = nn.Linear(config.n_embd, config.action_size, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4, bias=False),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, action_mask=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        logits = self.policy_head(pooled)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        policy = F.log_softmax(logits, dim=-1)
        value = self.value_head(pooled)
        return policy, value

    def predict(self, x, action_mask=None):
        self.eval()
        with torch.no_grad():
            log_p, v = self.forward(x, action_mask)
            return log_p.exp(), v

    def count_parameters(self) -> int:
        return self.n_params
