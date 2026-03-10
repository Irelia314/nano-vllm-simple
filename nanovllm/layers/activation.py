import torch
from torch import nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按最后一个维度分片
        x, y = x.chunk(2, -1)
        # silu加乘法门控
        return F.silu(x) * y