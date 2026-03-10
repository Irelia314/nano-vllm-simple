import torch
from torch import nn
import torch.nn.functional as F
from typing import List

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """默认权重加载器"""
        param.data.copy_(loaded_weight) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features_list: List[int],
        bias: bool = False,
    ) -> None:
        self.out_features_list = out_features_list
        super().__init__(in_features, sum(out_features_list), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int):
        """默认权重加载器"""
        shard_offset = sum(self.out_features_list[:shard_id])
        shard_size = self.out_features_list[shard_id]
        param_data = param.data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class QKVParallelLinear(Linear):
    def __init__(
        self,
        in_features: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.q_size = num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        out_features = self.q_size + 2 * self.kv_size
        super().__init__(in_features, out_features, bias)

        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str):
        param_data = param.data
        assert shard_id in ["q", "k", "v"]
        if shard_id == "q":
            shard_offset = 0
            shard_size = self.q_size
        elif shard_id == "k":
            shard_offset = self.q_size
            shard_size = self.kv_size
        else:  # shard_id == "v"
            shard_offset = self.q_size + self.kv_size
            shard_size = self.kv_size
        param_data = param.data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)
