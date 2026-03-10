import torch
from transformers import AutoTokenizer
import os
from nanovllm.config import Config
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.sampling_params import SamplingParams


class SimpleLLM:
    """最简版LLM推理引擎"""

    def __init__(self, model_path: str):
        # 加载配置
        self.config = Config(model_path)

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.eos_token_id = self.tokenizer.eos_token_id

        # 加载模型
        self.model = Qwen3ForCausalLM(self.config.hf_config)
        load_model(self.model, model_path)

        # 移到GPU
        self.model.cuda()
        self.model.eval()

    def _init_kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        hf_config = self.config.hf_config
        num_layers = hf_config.num_hidden_layers
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        max_len = self.config.max_model_len
        param = next(self.model.parameters())
        device = param.device
        dtype = param.dtype
        kv_caches = []
        for _ in range(num_layers):
            k_cache = torch.empty(max_len, num_kv_heads, head_dim, device=device, dtype=dtype)
            v_cache = torch.empty(max_len, num_kv_heads, head_dim, device=device, dtype=dtype)
            kv_caches.append((k_cache, v_cache))
        return kv_caches

    def generate(
        self, 
        prompt: str, 
        params: SamplingParams | None = None,
    ) -> str:
        """生成文本"""
        if params is None:
            params = SamplingParams()

        # 编码prompt
        tokens = self.tokenizer.encode(prompt)

        # 初始化KV Cache
        kv_caches = self._init_kv_cache()

        # 自回归生成
        with torch.inference_mode():
            if params.max_tokens <= 0:
                return self.tokenizer.decode(tokens)

            device = next(self.model.parameters()).device
            # prefill
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            positions = torch.arange(len(tokens), dtype=torch.long, device=device)
            hidden_states = self.model(
                input_ids,
                positions,
                kv_caches=kv_caches,
                cache_len=0,
                is_prefill=True,
            )
            logits = self.model.compute_logits(hidden_states)
            next_token = self._sample(logits[0, -1], params.temperature)
            tokens.append(next_token)

            if not params.ignore_eos and next_token == self.eos_token_id:
                return self.tokenizer.decode(tokens)

            cache_len = len(tokens) - 1
            for _ in range(params.max_tokens - 1):
                input_ids = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                positions = torch.tensor([cache_len], dtype=torch.long, device=device)
                hidden_states = self.model(
                    input_ids,
                    positions,
                    kv_caches=kv_caches,
                    cache_len=cache_len,
                    is_prefill=False,
                )
                logits = self.model.compute_logits(hidden_states)
                next_token = self._sample(logits[0, -1], params.temperature)
                tokens.append(next_token)
                cache_len += 1

                if not params.ignore_eos and next_token == self.eos_token_id:
                    break

        # 解码
        return self.tokenizer.decode(tokens)

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        """从logits采样一个token"""
        if temperature <= 1e-10:
            # 贪婪采样
            return logits.argmax(dim=-1).item()
        else:
            # 温度采样
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).item()


# 使用示例
if __name__ == "__main__":
    model_path = os.path.expanduser("/data/huangfangjie/qwen3-4b-thinking/")
    llm = SimpleLLM(model_path)

    params = SamplingParams(temperature=0.8, max_tokens=100)
    output = llm.generate("Hello, how are you?", params)
    # llm.verify_kv_cache("Hello, how are you?", steps=2)
    print(f"Output: {output}")
