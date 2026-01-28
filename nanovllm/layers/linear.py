import torch
from torch import nn
import torch.nn.functional as F


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载器"""
    param.data.copy_(loaded_weight)


class Linear(nn.Module):
    def __init__(
        self,
        in_features:int,
        out_features:int,
        bias:bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.weight_loader = default_weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.weight_loader = default_weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedLinear(nn.Module):
    """合并多个Linear的层，用于QKV或gate_up合并"""
    def __init__(
        self,
        in_features: int,
        out_features_list: list[int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features_list = out_features_list
        self.total_out_features = sum(out_features_list)

        self.weight = nn.Parameter(torch.empty(self.total_out_features, in_features))
        self.weight.weight_loader = self.merged_weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_out_features))
            self.bias.weight_loader = self.merged_weight_loader
        else:
            self.register_parameter("bias", None)

    def merged_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int):
        """加载合并权重的一部分"""
        # 计算当前shard的偏移量和大小
        shard_offset = sum(self.out_features_list[:shard_id])
        shard_size = self.out_features_list[shard_id]

        # 复制到对应位置
        param_data = param.data
        param_data[shard_offset:shard_offset + shard_size] = loaded_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)