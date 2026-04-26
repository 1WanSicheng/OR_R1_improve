import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TEMPLATE_Q2MC_EN = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
""".strip()


FORMAT_MARKERS = [
    "## Mathematical Model:",
    "## Decision Variables:",
    "## Objective Function:",
    "## Constraints:",
    "## Python Code Solution Using `coptpy`:",
    "```python",
]


ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best solution: {model.objval}")\nelse:\n    print("No Best Solution")'


def load_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def compile_script(script_content: str, timeout: int = 120):
    target_dir = "./eval_execute"
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir=target_dir, mode="w", encoding="utf-8") as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(script_content)

    try:
        process = subprocess.run(
            ["python", tmp_file_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=True,
        )
        execution_result = process.stdout
        start = execution_result.find("Just print the best solution:")
        if start != -1:
            execution_best_solution = execution_result[start:].replace("Just print the best solution:", "").strip()
            end = execution_best_solution.find("\n")
            if end != -1:
                execution_best_solution = execution_best_solution[:end]
            execution_state = "Execution Successful and Best Solution Found"
        elif "No Best Solution" in execution_result:
            execution_best_solution = "No Best Solution"
            execution_state = "Execution Successful but No Best Solution Found"
        else:
            execution_best_solution = None
            execution_state = "Execution Successful but Out of Expectation"
    except subprocess.TimeoutExpired as e:
        execution_result = (e.stdout or "") + "\n" + (e.stderr or "")
        execution_best_solution = None
        execution_state = "Execution Failed: Timeout"
    except subprocess.CalledProcessError as e:
        execution_result = (e.stdout or "") + "\n" + (e.stderr or "")
        execution_best_solution = None
        execution_state = f"Execution Failed: returncode={e.returncode}"
    finally:
        os.remove(tmp_file_name)

    return {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution,
        "execution_state": execution_state,
    }


def run_code(output: str):
    start = output.find("```python")
    if start == -1:
        return None
    end = output.find("```", start + 9)
    if end == -1:
        return None
    script = output[start:end].replace("```python", "") + ADD_SCRIPT
    out = compile_script(script)
    out["script"] = script
    return out


def safe_int_vote(prediction):
    try:
        return int(float(prediction))
    except Exception:
        return None


def relative_match(prediction, answer, tolerance=0.05):
    try:
        gt = float(answer)
        pred = float(prediction)
    except Exception:
        return False
    if gt == 0:
        return abs(pred) <= tolerance
    return abs((pred - gt) / gt) <= tolerance


def format_reward_of(text: str):
    hit = sum(marker in text for marker in FORMAT_MARKERS)
    return hit / len(FORMAT_MARKERS), [marker for marker in FORMAT_MARKERS if marker in text]


def build_sft_trace(tokenizer, sample: dict):
    prompt = sample["prompt"]
    completion = sample["completion"]
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        tokenize=False,
    ).replace("<think>\n\n</think>\n\n", "")
    full_ids = tokenizer(chat_text, return_tensors="pt").input_ids[0]
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return {
        "prompt_chars": len(prompt),
        "completion_chars": len(completion),
        "chat_text_chars": len(chat_text),
        "prompt_tokens": int(prompt_ids.shape[0]),
        "full_tokens": int(full_ids.shape[0]),
        "completion_tokens_effective": int(full_ids.shape[0] - prompt_ids.shape[0]),
        "prompt_preview": prompt[:1600],
        "completion_preview": completion[:2200],
    }


def build_grpo_prompt(tokenizer, sample: dict):
    prompt_body = TEMPLATE_Q2MC_EN.replace("{Question}", sample["question"].strip()).strip()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_body}], tokenize=False)
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return prompt_body, prompt, int(prompt_tokens.shape[0])


def trace_grpo(model_path: str, tokenizer, sample: dict, args):
    prompt_body, prompt, prompt_tokens = build_grpo_prompt(tokenizer, sample)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</s>", "<|endoftext|>", "<|im_end|>"],
    )
    generation = llm.generate([prompt], sampling)[0]
    completions = [out.text for out in generation.outputs]

    rows = []
    prediction_answers = []
    for idx, completion in enumerate(completions):
        fr, markers_hit = format_reward_of(completion)
        exec_output = run_code(completion)
        valid_code_reward = 0.0
        prediction = None
        if exec_output is not None:
            prediction = exec_output["execution_best_solution"]
            if prediction is not None:
                valid_code_reward = 1.0
        prediction_answers.append(prediction)
        rows.append(
            {
                "completion_index": idx,
                "completion_chars": len(completion),
                "completion_preview": completion[:2200],
                "format_reward": fr,
                "format_markers_hit": markers_hit,
                "valid_code_reward": valid_code_reward,
                "prediction_answer": prediction,
                "execution_state": None if exec_output is None else exec_output["execution_state"],
                "execution_result_tail": None
                if exec_output is None
                else exec_output["execution_result"][-1200:],
                "script_preview": None if exec_output is None else exec_output["script"][:2200],
            }
        )

    prediction_answer_dict = {}
    for prediction in prediction_answers:
        if prediction is None or prediction == "No Best Solution":
            continue
        voted = safe_int_vote(prediction)
        if voted is None:
            continue
        prediction_answer_dict[voted] = prediction_answer_dict.get(voted, 0) + 1

    voting_answer = None
    max_count = 1
    for key, count in prediction_answer_dict.items():
        if count > max_count:
            max_count = count
            voting_answer = key

    for row in rows:
        prediction = row["prediction_answer"]
        if prediction is None or prediction == "No Best Solution" or voting_answer is None:
            answer_reward = 0.0
        else:
            voted = safe_int_vote(prediction)
            answer_reward = 1.0 if voted == voting_answer else 0.0
        row["answer_reward"] = answer_reward
        row["total_reward"] = row["format_reward"] + row["valid_code_reward"] + answer_reward
        row["gt_answer"] = sample["answer"]
        row["matches_gt"] = relative_match(prediction, sample["answer"])

    return {
        "question_preview": sample["question"][:2400],
        "gt_answer": sample["answer"],
        "prompt_body_preview": prompt_body[:2200],
        "chat_prompt_preview": prompt[:2200],
        "prompt_tokens": prompt_tokens,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "voting_answer": voting_answer,
        "voting_counts": prediction_answer_dict,
        "rows": rows,
    }


def write_markdown(report_path: Path, sft_trace: dict, grpo_trace: dict, model_path: str, sft_path: str, grpo_path: str):
    lines = []
    lines.append("# OR-R1 Real Training Flow Trace")
    lines.append("")
    lines.append("## Run Setup")
    lines.append(f"- model: `{model_path}`")
    lines.append(f"- sft_dataset: `{sft_path}`")
    lines.append(f"- grpo_dataset: `{grpo_path}`")
    lines.append("")
    lines.append("## SFT Sample")
    lines.append(f"- prompt_chars: {sft_trace['prompt_chars']}")
    lines.append(f"- completion_chars: {sft_trace['completion_chars']}")
    lines.append(f"- prompt_tokens: {sft_trace['prompt_tokens']}")
    lines.append(f"- full_tokens: {sft_trace['full_tokens']}")
    lines.append(f"- completion_tokens_effective: {sft_trace['completion_tokens_effective']}")
    lines.append("")
    lines.append("### Prompt Preview")
    lines.append("```text")
    lines.append(sft_trace["prompt_preview"])
    lines.append("```")
    lines.append("")
    lines.append("### Completion Preview")
    lines.append("```text")
    lines.append(sft_trace["completion_preview"])
    lines.append("```")
    lines.append("")
    lines.append("## GRPO Sample")
    lines.append(f"- gt_answer: {grpo_trace['gt_answer']}")
    lines.append(f"- prompt_tokens: {grpo_trace['prompt_tokens']}")
    lines.append(f"- num_generations: {grpo_trace['num_generations']}")
    lines.append(f"- voting_answer: {grpo_trace['voting_answer']}")
    lines.append(f"- voting_counts: {json.dumps(grpo_trace['voting_counts'], ensure_ascii=False)}")
    lines.append("")
    lines.append("### Question Preview")
    lines.append("```text")
    lines.append(grpo_trace["question_preview"])
    lines.append("```")
    lines.append("")
    lines.append("### Prompt Preview")
    lines.append("```text")
    lines.append(grpo_trace["chat_prompt_preview"])
    lines.append("```")
    lines.append("")
    lines.append("### Reward Table")
    lines.append("")
    lines.append("| idx | pred | exec_state | format | valid_code | answer | total | gt_match |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for row in grpo_trace["rows"]:
        pred = row["prediction_answer"]
        pred_show = str(pred) if pred is not None else "None"
        exec_state = row["execution_state"] or "no_code"
        lines.append(
            f"| {row['completion_index']} | {pred_show} | {exec_state} | {row['format_reward']:.3f} | {row['valid_code_reward']:.1f} | {row['answer_reward']:.1f} | {row['total_reward']:.3f} | {row['matches_gt']} |"
        )
    lines.append("")
    for row in grpo_trace["rows"]:
        lines.append(f"### Completion {row['completion_index']}")
        lines.append(f"- prediction_answer: {row['prediction_answer']}")
        lines.append(f"- execution_state: {row['execution_state']}")
        lines.append(f"- format_markers_hit: {json.dumps(row['format_markers_hit'], ensure_ascii=False)}")
        lines.append(f"- total_reward: {row['total_reward']:.3f}")
        lines.append("")
        lines.append("```text")
        lines.append(row["completion_preview"] or "")
        lines.append("```")
        lines.append("")
        if row["script_preview"]:
            lines.append("```python")
            lines.append(row["script_preview"])
            lines.append("```")
            lines.append("")
        if row["execution_result_tail"]:
            lines.append("```text")
            lines.append(row["execution_result_tail"])
            lines.append("```")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--sft_dataset_path", required=True)
    parser.add_argument("--grpo_dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sft_index", type=int, default=0)
    parser.add_argument("--grpo_index", type=int, default=98)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.72)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    sft_dataset = load_json_or_jsonl(args.sft_dataset_path)
    sft_sample = sft_dataset[args.sft_index]
    sft_trace = build_sft_trace(tokenizer, sft_sample)

    grpo_dataset = load_json_or_jsonl(args.grpo_dataset_path)
    grpo_sample = grpo_dataset[args.grpo_index]
    grpo_trace = trace_grpo(args.model_name_or_path, tokenizer, grpo_sample, args)

    raw_path = output_dir / "or_r1_flow_trace.json"
    report_path = output_dir / "or_r1_flow_trace_report.md"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sft_trace": sft_trace,
                "grpo_trace": grpo_trace,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    write_markdown(report_path, sft_trace, grpo_trace, args.model_name_or_path, args.sft_dataset_path, args.grpo_dataset_path)
    print(json.dumps({"raw_trace": str(raw_path), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
