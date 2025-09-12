import os
import math
from typing import Optional

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


class LLMWrapper:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None, n_gpu_layers: int = -1):
        if Llama is None:
            raise RuntimeError("llama-cpp-python failed to import. Install a compatible wheel: CPU 'pip install llama-cpp-python[openblas]' or CUDA 'pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 llama-cpp-python'.")
        if n_threads is None:
            try:
                import multiprocessing
                n_threads = max(1, multiprocessing.cpu_count())
            except Exception:
                n_threads = 1
        try:
            self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads, logits_all=False, seed=0)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize llama-cpp. {e}")

    def tokenize_count(self, text: str) -> int:
        ids = self.model.tokenize(text.encode("utf-8"))
        return len(ids)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, repeat_penalty: float, seed: int) -> str:
        out = self.model(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=int(seed),
            echo=False,
            stop=None,
        )
        txt = out["choices"][0]["text"]
        return txt.strip()


class MockLLM:
    def __init__(self):
        pass

    def tokenize_count(self, text: str) -> int:
        return int(math.ceil(len(text.strip().split()) * 1.5))

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, repeat_penalty: float, seed: int) -> str:
        import random
        base = int(seed) ^ (max_new_tokens << 8) ^ (int(temperature * 1000) << 16) ^ (int(top_p * 1000) << 24) ^ (int(repeat_penalty * 1000) << 2)
        rng = random.Random(base)
        words = [w for w in prompt.strip().split() if w.isalpha()]
        base = words[-20:] if words else ["C1"]
        vocab = ["because", "we", "agree", "name", "is", "clear", "choose", "symbol"]
        k = max(1, min(max_new_tokens, 20))
        seq = []
        for _ in range(k):
            if rng.random() < 0.3:
                seq.append(rng.choice(base))
            else:
                seq.append(rng.choice(vocab))
        return " ".join(seq)
