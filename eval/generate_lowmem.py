import argparse
import json
import os

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TEMPLATE_q2mc_en = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
"""

OUTPUT_TEMPLATE = r"""

Fill in the following format:
## Mathematical Model:
YOUR ANSWER HERE

## Decision Variables:
YOUR ANSWER HERE

## Objective Function:
YOUR ANSWER HERE

## Constraints:
YOUR ANSWER HERE

## Python Code Solution Using `coptpy`:
```python
YOUR ANSWER HERE
```
"""


def main(args):
    assert os.path.exists(args.model_name_or_path), "We only support local model path!"
    os.makedirs(args.save_dir, exist_ok=True)
    save_file = os.path.join(args.save_dir, "generated.jsonl")
    if os.path.exists(save_file):
        print(f"File {save_file} already exists. Exiting to avoid overwriting.")
        return

    if not args.dataset_name.endswith(("jsonl", "json")):
        ds = datasets.load_dataset(args.dataset_name)[args.dataset_split]
    else:
        ds = datasets.load_dataset("json", data_files=args.dataset_name)["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    samples = []
    for example in ds:
        if "en_question" in example:
            prompt_body = TEMPLATE_q2mc_en.replace("{Question}", example["en_question"].strip()).strip()
        else:
            prompt_body = TEMPLATE_q2mc_en.replace("{Question}", example["question"].strip()).strip()

        if "BASELINE" in args.model_name_or_path:
            prompt_body = prompt_body + OUTPUT_TEMPLATE

        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_body}], tokenize=False)
        item = {k: v for k, v in example.items() if k != "prompt"}
        item["prompt"] = prompt
        samples.append(item)

    print(f"load dataset from `{args.dataset_name}` done. sample size: {len(samples)}")

    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )
    print("init model done.")

    stop_tokens = ["</s>", "<|endoftext|>", "<|im_end|>"]
    if args.decoding_method == "greedy":
        sampling_params = SamplingParams(
            n=args.topk,
            temperature=0,
            top_p=1,
            max_tokens=args.max_tokens,
            stop=stop_tokens,
        )
    else:
        sampling_params = SamplingParams(
            n=args.topk,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=stop_tokens,
        )

    prompts = [sample["prompt"] for sample in samples]
    generations = model.generate(prompts, sampling_params)

    with open(save_file, "w", encoding="utf-8") as fw:
        for sample, prompt, generation in zip(samples, prompts, generations):
            touched_output = set()
            for output in generation.outputs:
                text = output.text
                if text in touched_output:
                    continue
                touched_output.add(text)
                example_t = {k: v for k, v in sample.items()}
                example_t["q2mc_en_prompt"] = prompt
                example_t["en_math_model_coptpy_code"] = text
                if args.verbose:
                    print("-" * 20 + "prompt" + "-" * 20)
                    print(prompt)
                    print("-" * 20 + "completion" + "-" * 20)
                    print(text)
                    print("-" * 80)
                fw.write(json.dumps(example_t, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--max_num_seqs", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.72)
    parser.add_argument("--decoding_method", type=str, default=None)
    parser.add_argument("--decoding", dest="decoding_method", type=str, default=None)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.decoding_method is None:
        args.decoding_method = "greedy"
    return args


if __name__ == "__main__":
    main(parse_args())
