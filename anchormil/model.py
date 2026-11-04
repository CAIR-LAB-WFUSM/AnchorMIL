"""AnchorMIL: anchored regression + classification MIL for Oncotype DX prediction.

Follows Sec. 3 of Koyun et al., "H&E to Oncotype DX: Predicting recurrence risk in
HR+/HER2- breast cancer" (Expert Systems With Applications 303, 2026, 130469).
"""
from typing import Dict, Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two linear layers with GELU in between (Sec. 3.4, Feed Forward)."""

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Q = LN(A) W_Q ; [K, V] = LN(X) W_KV ; A* = A + MHA(Q, K, V)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.w_q = nn.Linear(dim, dim)
        self.w_kv = nn.Linear(dim, 2 * dim)
        self.w_o = nn.Linear(dim, dim)
        self.p = dropout

    def forward(self, anchors, x, pad_mask: Optional[torch.Tensor] = None):
        B, K, d = anchors.shape
        N = x.shape[1]
        q = self.w_q(self.ln_q(anchors)).view(B, K, self.heads, self.dh).transpose(1, 2)
        k, v = self.w_kv(self.ln_kv(x)).chunk(2, dim=-1)
        k = k.reshape(B, N, self.heads, self.dh).transpose(1, 2)
        v = v.reshape(B, N, self.heads, self.dh).transpose(1, 2)
        attn_mask = None
        if pad_mask is not None:  # pad_mask: (B, N), True = padded
            attn_mask = (~pad_mask)[:, None, None, :]  # True = attend
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, K, d)
        return anchors + self.w_o(out)


class SelfAttention(nn.Module):
    """[Q, K, V] = LN(A*) W_QKV ; A~ = A* + MHA(Q, K, V)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.ln = nn.LayerNorm(dim)
        self.w_qkv = nn.Linear(dim, 3 * dim)
        self.w_o = nn.Linear(dim, dim)
        self.p = dropout

    def forward(self, a):
        B, K, d = a.shape
        qkv = self.w_qkv(self.ln(a)).view(B, K, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0
        )
        return a + self.w_o(out.transpose(1, 2).reshape(B, K, d))


class DecoderBlock(nn.Module):
    """Cross-attention -> self-attention -> MLP, all residual.

    The cross-attention module is injected so its weights can be shared
    across blocks (Perceiver-style, Fig. 2).
    """

    def __init__(self, dim, heads, dropout, cross: CrossAttention):
        super().__init__()
        self.cross = cross
        self.self_attn = SelfAttention(dim, heads, dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.ln_ff = nn.LayerNorm(dim)

    def forward(self, anchors, x, pad_mask=None):
        a = self.cross(anchors, x, pad_mask)
        a = self.self_attn(a)
        return a + self.ff(self.ln_ff(a))


class AnchorMIL(nn.Module):
    """
    Args:
        in_dim: patch feature dimension (768 for CTransPath).
        num_anchors: K learnable anchors (128 in the paper).
        score_range: (low, high) ODX interval over which fixed anchor scores s_k are spaced.
        share_cross_attn: share cross-attention weights across blocks.
    """

    def __init__(
        self,
        in_dim: int = 768,
        dim: int = 768,
        num_anchors: int = 128,
        num_heads: int = 12,
        num_blocks: int = 2,
        dropout: float = 0.1,
        score_range=(0.0, 100.0),
        share_cross_attn: bool = True,
    ):
        super().__init__()
        self.proj = nn.Identity() if in_dim == dim else nn.Linear(in_dim, dim)
        self.anchors = nn.Parameter(torch.randn(num_anchors, dim) * 0.02)
        # fixed anchor scores s_k, evenly spaced over the ODX interval
        self.register_buffer("anchor_scores", torch.linspace(score_range[0], score_range[1], num_anchors))

        shared = CrossAttention(dim, num_heads, dropout) if share_cross_attn else None
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(dim, num_heads, dropout, shared if share_cross_attn else CrossAttention(dim, num_heads, dropout))
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.h_c = nn.Linear(dim, 1)  # confidence
        self.h_r = nn.Linear(dim, 1)  # regression offset
        self.h_p = nn.Linear(dim, 1)  # classification logit
        nn.init.zeros_(self.h_r.weight)
        nn.init.zeros_(self.h_r.bias)

    def forward(self, feats: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        feats: (B, N, in_dim) patch features; pad_mask: (B, N) bool, True for padding.
        Returns per-anchor tensors (B, K) and aggregated predictions (B,).
        """
        x = self.proj(feats)
        a = self.anchors.unsqueeze(0).expand(x.shape[0], -1, -1)
        for blk in self.blocks:
            a = blk(a, x, pad_mask)
        a = self.norm(a)

        c = torch.softmax(self.h_c(a).squeeze(-1), dim=-1)              # c_k, sums to 1
        r = self.anchor_scores.unsqueeze(0) + self.h_r(a).squeeze(-1)   # r_k = s_k + h_r(a_k)
        p = self.h_p(a).squeeze(-1)                                     # p_k (logit)

        r_hat = (c * r).sum(-1)   # continuous ODX prediction
        p_hat = (c * p).sum(-1)   # aggregated logit
        return {"r": r, "p": p, "c": c, "reg": r_hat, "logit": p_hat, "prob": torch.sigmoid(p_hat)}
