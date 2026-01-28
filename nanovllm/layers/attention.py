import torch
from torch import nn
import math


class Attention(nn.Module):
    """简单的注意力实现，暂不用flash-attn"""
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        输入形状: [seq_len, num_heads, head_dim]
        输出形状: [seq_len, num_heads * head_dim]
        """
        seq_len, num_heads, head_dim = q.shape

        # 计算注意力分数: [seq_len, seq_len, num_heads]
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.scale

        # causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

        # softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和: [num_heads, seq_len, head_dim]
        output = torch.einsum("hqk,khd->qhd", attn_weights, v)

        # reshape: [seq_len, num_heads * head_dim]
        return output.flatten(1, -1)