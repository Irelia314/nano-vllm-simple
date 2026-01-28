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

        # 自回归生成
        with torch.inference_mode():
            for _ in range(params.max_tokens):
                # 准备输入
                input_ids = torch.tensor([tokens], dtype=torch.long).cuda()
                positions = torch.arange(len(tokens), dtype=torch.long).cuda()

                # 前向传播
                hidden_states = self.model(input_ids, positions)
                logits = self.model.compute_logits(hidden_states)

                # 采样下一个token
                next_token = self._sample(logits[0, -1], params.temperature)
                tokens.append(next_token)

                # 检查是否结束
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
    model_path = os.path.expanduser("~/qwen3-4b-thinking/")
    llm = SimpleLLM(model_path)

    params = SamplingParams(temperature=0.8, max_tokens=100)
    output = llm.generate("Hello, how are you?", params)

    print(f"Output: {output}")