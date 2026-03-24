from transformers import pipeline
import torch

# 🔥 CPU optimization
torch.set_num_threads(4)

_pipe = None  # global cache


class LocalLLM:
    def __init__(self):
        global _pipe
        if _pipe is None:
            _pipe = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_new_tokens=60,   # 🔥 reduced
                do_sample=False,
                return_full_text=False
            )
        self.pipe = _pipe

    def invoke(self, prompt: str):
        result = self.pipe(prompt)[0]["generated_text"]
        return result.strip()


def get_llm():
    return LocalLLM()