import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TEMPLATE_Q2MC_EN = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
""".strip()

REPAIR_TEMPLATE = r"""
You previously generated an operations research formulation and `coptpy` solver code for the question below.

# Question:
{question}

# Previous Response:
{previous_response}

# Execution and Modeling Feedback:
{feedback}

Repair the response with targeted changes only. Make the mathematical model and the `coptpy` code consistent, executable, and closer to the correct optimization problem.

Return the full corrected answer in exactly this format:
## Mathematical Model:
...

## Decision Variables:
...

## Objective Function:
...

## Constraints:
...

## Python Code Solution Using `coptpy`:
```python
...
```
""".strip()

ADD_SCRIPT = r'''
try:
    if "model" not in globals():
        print("TRAJECTORY_STATUS: missing_model_variable")
    elif model.status == COPT.OPTIMAL:
        print(f"Just print the best solution: {model.objval}")
        print("TRAJECTORY_STATUS: optimal")
    else:
        print("No Best Solution")
        print(f"TRAJECTORY_STATUS: non_optimal_status_{model.status}")
except NameError as exc:
    print(f"TRAJECTORY_STATUS: missing_variable: {exc}")
except Exception as exc:
    print(f"TRAJECTORY_STATUS: post_solve_error: {type(exc).__name__}: {exc}")
'''


@dataclass
class ExecutionResult:
    state: str
    best_solution: str | None
    stdout: str
    stderr: str
    script: str | None


def extract_code(text: str) -> str | None:
    start = text.find("```python")
    if start == -1:
        return None
    end = text.find("```", start + len("```python"))
    if end == -1:
        return None
    code = text[start + len("```python"):end].strip()
    return code or None


def run_code(response: str, timeout: int) -> ExecutionResult:
    code = extract_code(response)
    if not code:
        return ExecutionResult("no_code", None, "", "", None)

    script = code + "\n" + ADD_SCRIPT
    os.makedirs("./eval_execute", exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir="./eval_execute", mode="w", encoding="utf-8") as fd:
        path = fd.name
        fd.write(script)

    try:
        proc = subprocess.run(
            ["python", path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        best = parse_best_solution(stdout)
        state = classify_execution(proc.returncode, stdout, stderr, best)
        return ExecutionResult(state, best, stdout, stderr, script)
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult("timeout", None, exc.stdout or "", exc.stderr or "", script)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def parse_best_solution(stdout: str) -> str | None:
    marker = "Just print the best solution:"
    pos = stdout.find(marker)
    if pos == -1:
        return None
    answer = stdout[pos + len(marker):].strip().splitlines()[0].strip()
    return answer or None


def classify_execution(returncode: int, stdout: str, stderr: str, best: str | None) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if best is not None:
        return "optimal"
    if returncode != 0:
        if "syntaxerror" in text:
            return "syntax_error"
        if "nameerror" in text or "missing_variable" in text or "missing_model_variable" in text:
            return "missing_variable"
        return "runtime_error"
    if "unbounded" in text:
        return "unbounded"
    if "infeasible" in text:
        return "infeasible"
    if "no best solution" in text:
        return "non_optimal"
    return "unexpected_no_solution"


def numeric_distance(prediction: str | None, answer: Any) -> float | None:
    if prediction is None or prediction == "No Best Solution":
        return None
    try:
        gt = float(answer)
        pred = float(prediction)
    except (TypeError, ValueError):
        return None
    if gt == 0:
        return abs(pred)
    return abs((pred - gt) / gt)


def is_correct(prediction: str | None, answer: Any, tolerance: float) -> bool:
    if answer == "No Best Solution":
        return prediction == answer
    dist = numeric_distance(prediction, answer)
    return dist is not None and dist <= tolerance


def infer_omission_feedback(question: str, response: str) -> list[str]:
    checks = [
        ("capacity", ["capacity", "limit", "available", "at most", "no more than", "budget"]),
        ("demand", ["demand", "require", "at least", "satisfy", "minimum"]),
        ("assignment uniqueness", ["assign", "choose", "select", "only one", "at most one"]),
        ("inventory balance", ["inventory", "storage", "beginning", "end of", "period", "month", "quarter"]),
        ("flow balance", ["flow", "origin", "destination", "supply", "shipment", "transport"]),
        ("integrality/binary", ["integer", "binary", "yes/no", "whether", "choose", "select"]),
        ("time consistency", ["time", "week", "period", "shift", "schedule"]),
    ]
    q = question.lower()
    r = response.lower()
    missing = []
    for label, triggers in checks:
        if any(token in q for token in triggers) and not any(token in r for token in triggers[:3]):
            missing.append(label)
    return missing


def build_feedback(question: str, response: str, result: ExecutionResult, answer: Any) -> str:
    lines = [f"- Execution state: {result.state}"]
    if result.best_solution is not None:
        lines.append(f"- Solver objective value: {result.best_solution}")
        dist = numeric_distance(result.best_solution, answer)
        if dist is not None:
            lines.append(f"- Relative distance to reference answer: {dist:.6g}")
    else:
        lines.append("- No valid objective value was produced.")

    if result.state in {"syntax_error", "runtime_error", "missing_variable", "timeout"}:
        err = (result.stderr or result.stdout).strip()
        err = "\n".join(err.splitlines()[-8:])
        if err:
            lines.append("- Error excerpt:")
            lines.append(err)
    if result.state in {"infeasible", "unbounded", "non_optimal", "unexpected_no_solution"}:
        lines.append("- The solver did not return an optimal solution. Check objective direction, variable domains, and constraint coverage.")

    missing = infer_omission_feedback(question, response)
    if missing:
        lines.append("- Suspicious constraint omissions: " + ", ".join(missing))
    lines.append("- Repair focus: keep correct parts, fix only the likely cause, and ensure the final code defines and solves `model`.")
    return "\n".join(lines)


def apply_chat(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False)


def load_samples(args):
    if args.dataset_name.endswith(("jsonl", "json")):
        ds = datasets.load_dataset("json", data_files=args.dataset_name)["train"]
    else:
        ds = datasets.load_dataset(args.dataset_name)[args.dataset_split]
    samples = list(ds)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    return samples


def summarize(rows: list[dict], tolerance: float) -> dict:
    total = len(rows)
    initial_correct = sum(is_correct(r["initial_execution_best_solution"], r["answer"], tolerance) for r in rows)
    repaired_correct = sum(is_correct(r["repaired_execution_best_solution"], r["answer"], tolerance) for r in rows)
    initial_optimal = sum(r["initial_execution_state"] == "optimal" for r in rows)
    repaired_optimal = sum(r["repaired_execution_state"] == "optimal" for r in rows)
    repaired_improved = 0
    for row in rows:
        before = numeric_distance(row["initial_execution_best_solution"], row["answer"])
        after = numeric_distance(row["repaired_execution_best_solution"], row["answer"])
        if before is None and after is not None:
            repaired_improved += 1
        elif before is not None and after is not None and after < before:
            repaired_improved += 1
    return {
        "num_samples": total,
        "initial_pass@1": initial_correct / total if total else 0.0,
        "repaired_pass@1": repaired_correct / total if total else 0.0,
        "initial_optimal_rate": initial_optimal / total if total else 0.0,
        "repaired_optimal_rate": repaired_optimal / total if total else 0.0,
        "repair_improved_rate": repaired_improved / total if total else 0.0,
    }


def main(args):
    assert os.path.exists(args.model_name_or_path), "Only local model paths are supported."
    os.makedirs(args.save_dir, exist_ok=True)
    output_file = os.path.join(args.save_dir, "trajectory.jsonl")
    metrics_file = os.path.join(args.save_dir, "trajectory.metrics.json")
    if os.path.exists(metrics_file) and not args.overwrite:
        print(f"{metrics_file} already exists. Use --overwrite to rerun.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    samples = load_samples(args)
    print(f"Loaded {len(samples)} samples from {args.dataset_name}")

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0,
        top_p=1,
        max_tokens=args.max_tokens,
        stop=["</s>", "<|endoftext|>", "<|im_end|>"],
    )

    initial_prompts = []
    for sample in samples:
        question = sample.get("en_question") or sample["question"]
        initial_prompts.append(apply_chat(tokenizer, TEMPLATE_Q2MC_EN.replace("{Question}", question.strip())))
    initial_generations = llm.generate(initial_prompts, sampling)

    repair_prompts = []
    partial_rows = []
    for sample, prompt, generation in zip(samples, initial_prompts, initial_generations):
        question = sample.get("en_question") or sample["question"]
        initial_response = generation.outputs[0].text
        initial_result = run_code(initial_response, args.execution_timeout)
        feedback = build_feedback(question, initial_response, initial_result, sample["answer"])
        repair_body = REPAIR_TEMPLATE.format(
            question=question.strip(),
            previous_response=initial_response.strip(),
            feedback=feedback,
        )
        repair_prompts.append(apply_chat(tokenizer, repair_body))
        row = {k: v for k, v in sample.items()}
        row.update(
            {
                "initial_prompt": prompt,
                "initial_response": initial_response,
                "initial_execution_state": initial_result.state,
                "initial_execution_best_solution": initial_result.best_solution,
                "initial_feedback": feedback,
            }
        )
        partial_rows.append(row)

    repair_generations = llm.generate(repair_prompts, sampling)
    rows = []
    for row, repair_prompt, generation in zip(partial_rows, repair_prompts, repair_generations):
        repaired_response = generation.outputs[0].text
        repaired_result = run_code(repaired_response, args.execution_timeout)
        row.update(
            {
                "repair_prompt": repair_prompt,
                "repaired_response": repaired_response,
                "repaired_execution_state": repaired_result.state,
                "repaired_execution_best_solution": repaired_result.best_solution,
                "en_math_model_coptpy_code": repaired_response,
                "execution_state": repaired_result.state,
                "execution_best_solution": repaired_result.best_solution,
            }
        )
        rows.append(row)

    with open(output_file, "w", encoding="utf-8") as fw:
        for row in rows:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = summarize(rows, args.numerical_err_tolerance)
    with open(metrics_file, "w", encoding="utf-8") as fw:
        json.dump(metrics, fw, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--execution_timeout", type=int, default=180)
    parser.add_argument("--numerical_err_tolerance", type=float, default=0.05)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
