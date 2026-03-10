import torch
from torch import nn


class Attention(nn.Module):
    """简单的注意力实现，暂不用flash-attn，支持GQA"""
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        cache_len: int = 0,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        输入形状:
            q: [seq_len, num_q_heads, head_dim]
            k: [seq_len, num_kv_heads, head_dim]
            v: [seq_len, num_kv_heads, head_dim]
        输出形状: [seq_len, num_q_heads * head_dim]
        """
        seq_len, num_q_heads, head_dim = q.shape
        _, num_kv_heads, _ = k.shape

        # 写入KV Cache
        if k_cache is not None and v_cache is not None:
            if is_prefill:
                k_cache[: k.shape[0]].copy_(k)
                v_cache[: v.shape[0]].copy_(v)
                k_attn, v_attn = k, v
            else:
                k_cache[cache_len: cache_len + k.shape[0]].copy_(k)
                v_cache[cache_len: cache_len + v.shape[0]].copy_(v)
                k_attn = k_cache[: cache_len + k.shape[0]]
                v_attn = v_cache[: cache_len + v.shape[0]]
        else:
            k_attn, v_attn = k, v

        # GQA: 将 k/v 重复扩展以匹配 q 的 head 数量
        if num_kv_heads != num_q_heads:
            head_per_kv_head = num_q_heads // num_kv_heads
            k_attn = k_attn.repeat_interleave(head_per_kv_head, dim=1)  # [kv_len, num_q_heads, head_dim]
            v_attn = v_attn.repeat_interleave(head_per_kv_head, dim=1)  # [kv_len, num_q_heads, head_dim]

        # 计算注意力分数: [seq_len, seq_len, num_heads]
        scores = torch.einsum("qhd,khd->hqk", q, k_attn) * self.scale

        # causal mask（仅prefill时需要）
        if is_prefill:
            mask = torch.triu(
                torch.ones(seq_len, k_attn.shape[0], dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        # softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和: [num_heads, seq_len, head_dim]
        output = torch.einsum("hqk,khd->qhd", attn_weights, v_attn)

        # reshape: [seq_len, num_heads * head_dim]
        return output.flatten(1, -1)
