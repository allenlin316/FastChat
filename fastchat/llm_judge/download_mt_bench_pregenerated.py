"""
Download the pre-generated model answers and judgments for MT-bench.
"""
import os

from fastchat.utils import run_cmd

filenames = [
    "data/mt_bench/model_answer/llama2-bppo.jsonl",
    "data/mt_bench/model_answer/llama2-ours-outliers-bppo.jsonl",
    "data/mt_bench/model_answer/llama2-ours-outliers-qlora.jsonl",
    "data/mt_bench/model_answer/llama2-ours-outliers-smart.jsonl",
    "data/mt_bench/model_answer/llama2-qlora.jsonl",
    "data/mt_bench/model_answer/gpt-4.1-mini.jsonl",
    "data/mt_bench/model_judgment/gpt-4.1-mini_single.jsonl"
]

if __name__ == "__main__":
    prefix = "https://huggingface.co/spaces/allenlin316/mt-bench-gpt4_1/resolve/main/"

    for name in filenames:
        os.makedirs(os.path.dirname(name), exist_ok=True)
        ret = run_cmd(f"wget -q --show-progress -O {name} {prefix + name}")
        assert ret == 0