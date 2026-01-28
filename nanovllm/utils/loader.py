import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载器"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从safetensors文件加载模型权重"""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                # 检查是否需要权重名称映射
                param_name = weight_name
                shard_id = None
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, sid = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        shard_id = sid
                        break

                try:
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    loaded_weight = f.get_tensor(weight_name)
                    if shard_id is not None:
                        weight_loader(param, loaded_weight, shard_id)
                    else:
                        weight_loader(param, loaded_weight)
                except AttributeError:
                    # 如果找不到参数，跳过
                    continue