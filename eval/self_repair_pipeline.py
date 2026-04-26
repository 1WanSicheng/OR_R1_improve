import argparse
import json
import os
from pathlib import Path
from typing import Any

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from eval.generate_repair_lowmem import (
    REPAIR_TEMPLATE,
    TEMPLATE_Q2MC_EN,
    apply_chat,
    build_feedback,
    is_correct,
    numeric_distance,
    run_code,
)
from eval.analyze_errors import classify_wrong


DIAGNOSIS_REPAIR_TEMPLATE = r"""
You previously generated an operations research formulation and `coptpy` solver code for the question below.

# Question:
{question}

# Previous Response:
{previous_response}

# Structured Diagnosis:
failure_type: {failure_type}
diagnostic_tags: {diagnostic_tags}
likely_cause: {likely_cause}
repair_instruction: {repair_instruction}

# Execution Evidence:
{feedback}

Repair the response with targeted changes only. Do not rewrite unrelated parts. The final answer must define and solve `model` in `coptpy`.

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


SELF_REPAIR_SFT_PROMPT = r"""
You are improving a failed operations research solution.

# Problem
{question}

# Failed First-Pass Output
{failed_output}

# Diagnosis
failure_type: {failure_type}
diagnostic_tags: {diagnostic_tags}
likely_cause: {likely_cause}
repair_instruction: {repair_instruction}

Produce a corrected mathematical model and executable `coptpy` code.
""".strip()


def load_samples(path: str, split: str, max_samples: int | None):
    if path.endswith(("jsonl", "json")):
        ds = datasets.load_dataset("json", data_files=path)["train"]
    else:
        ds = datasets.load_dataset(path)[split]
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]
    return rows


def best_prediction(row: dict, prefix: str) -> Any:
    return row.get(f"{prefix}_execution_best_solution")


def diagnosis_from_row(row: dict, tolerance: float) -> dict:
    pseudo = {
        "question": row["question"],
        "answer": row["answer"],
        "en_math_model_coptpy_code": row["initial_response"],
        "execution_state": row["initial_execution_state"],
        "execution_best_solution": row["initial_execution_best_solution"],
        "execution_result": row.get("initial_execution_stdout", "") + "\n" + row.get("initial_execution_stderr", ""),
    }
    category, tags = classify_wrong(pseudo, tolerance)
    tags = list(dict.fromkeys(tags))
    cause_bits = []
    instruction_bits = []
    if category.startswith("execution/"):
        cause_bits.append("The generated code likely has an executable-code or API issue.")
        instruction_bits.append("Fix the code error while preserving the intended optimization model.")
    if "infeasible" in tags:
        cause_bits.append("The model is likely over-constrained or has a wrong constraint direction.")
        instruction_bits.append("Review constraint signs, lower/upper bounds, and feasibility of balance constraints.")
    if "unbounded" in tags:
        cause_bits.append("The model is likely missing bounds or resource constraints.")
        instruction_bits.append("Add missing non-negativity, upper bounds, capacity, or budget constraints.")
    if "missing_integrality_or_binary" in tags:
        cause_bits.append("A discrete selection/counting decision may have been modeled as continuous.")
        instruction_bits.append("Use binary or integer variables where the problem describes choices/counts.")
    if any(t.startswith("possible_missing_") for t in tags):
        missing = [t.replace("possible_missing_", "") for t in tags if t.startswith("possible_missing_")]
        cause_bits.append("Some key constraint categories may be missing: " + ", ".join(missing))
        instruction_bits.append("Add explicit constraints for the suspected missing categories.")
    if "objective_direction_mismatch" in tags:
        cause_bits.append("The objective direction may be inconsistent with the problem statement.")
        instruction_bits.append("Correct the objective sense and objective coefficients.")
    if "numeric_too_high" in tags:
        cause_bits.append("The predicted optimum is too high, often caused by missing limiting constraints in maximization or weak constraints.")
        instruction_bits.append("Check capacity, budget, upper-bound, mutual-exclusion, and demand constraints.")
    if "numeric_too_low" in tags:
        cause_bits.append("The predicted optimum is too low, often caused by over-restrictive constraints or objective coefficient mistakes.")
        instruction_bits.append("Check objective coefficients and whether constraints are stronger than stated.")
    if not cause_bits:
        cause_bits.append("The model is executable but does not match the target answer.")
        instruction_bits.append("Compare the formulation against the problem statement and correct structural modeling errors.")
    return {
        "failure_type": category,
        "diagnostic_tags": tags,
        "likely_cause": " ".join(cause_bits),
        "repair_instruction": " ".join(instruction_bits),
    }


def generate_texts(llm, prompts, max_tokens: int):
    sampling = SamplingParams(
        n=1,
        temperature=0,
        top_p=1,
        max_tokens=max_tokens,
        stop=["</s>", "<|endoftext|>", "<|im_end|>"],
    )
    gens = llm.generate(prompts, sampling)
    return [g.outputs[0].text for g in gens]


def summarize(rows: list[dict], tolerance: float, prefix: str) -> dict:
    total = len(rows)
    correct = 0
    optimal = 0
    feasible = 0
    improved = 0
    for row in rows:
        pred = row.get(f"{prefix}_execution_best_solution")
        state = row.get(f"{prefix}_execution_state")
        if is_correct(pred, row["answer"], tolerance):
            correct += 1
        if state == "optimal":
            optimal += 1
        if pred is not None and pred != "No Best Solution":
            feasible += 1
        if prefix != "initial":
            before = numeric_distance(row.get("initial_execution_best_solution"), row["answer"])
            after = numeric_distance(pred, row["answer"])
            if before is None and after is not None:
                improved += 1
            elif before is not None and after is not None and after < before:
                improved += 1
    return {
        f"{prefix}_num_samples": total,
        f"{prefix}_pass@1": correct / total if total else 0.0,
        f"{prefix}_optimal_rate": optimal / total if total else 0.0,
        f"{prefix}_feasible_answer_rate": feasible / total if total else 0.0,
        f"{prefix}_improved_rate": improved / total if total and prefix != "initial" else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_failed", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--execution_timeout", type=int, default=180)
    parser.add_argument("--numerical_err_tolerance", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trajectory_file = save_dir / "self_repair_trajectories.jsonl"
    metrics_file = save_dir / "self_repair_metrics.json"
    sft_file = save_dir / "self_repair_sft.jsonl"
    if metrics_file.exists() and not args.overwrite:
        print(f"{metrics_file} exists; use --overwrite to rerun.")
        return

    samples = load_samples(args.dataset_name, args.dataset_split, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )

    initial_prompts = []
    for sample in samples:
        question = sample.get("en_question") or sample["question"]
        initial_prompts.append(apply_chat(tokenizer, TEMPLATE_Q2MC_EN.replace("{Question}", question.strip())))
    initial_outputs = generate_texts(llm, initial_prompts, args.max_tokens)

    rows = []
    failed_rows = []
    for sample, prompt, output in zip(samples, initial_prompts, initial_outputs):
        question = sample.get("en_question") or sample["question"]
        result = run_code(output, args.execution_timeout)
        row = {k: v for k, v in sample.items()}
        row["question"] = question
        row.update(
            {
                "initial_prompt": prompt,
                "initial_response": output,
                "initial_execution_state": result.state,
                "initial_execution_best_solution": result.best_solution,
                "initial_execution_stdout": result.stdout,
                "initial_execution_stderr": result.stderr,
            }
        )
        if not is_correct(result.best_solution, row["answer"], args.numerical_err_tolerance):
            failed_rows.append(row)
        rows.append(row)

    failed_rows = failed_rows[: args.max_failed]
    for row in failed_rows:
        feedback = build_feedback(
            row["question"],
            row["initial_response"],
            type("Result", (), {
                "state": row["initial_execution_state"],
                "best_solution": row["initial_execution_best_solution"],
                "stdout": row.get("initial_execution_stdout", ""),
                "stderr": row.get("initial_execution_stderr", ""),
            })(),
            row["answer"],
        )
        row["coarse_feedback"] = feedback
        row["diagnosis"] = diagnosis_from_row(row, args.numerical_err_tolerance)

    coarse_prompts = [
        apply_chat(
            tokenizer,
            REPAIR_TEMPLATE.format(
                question=row["question"],
                previous_response=row["initial_response"],
                feedback=row["coarse_feedback"],
            ),
        )
        for row in failed_rows
    ]
    diagnosis_prompts = [
        apply_chat(
            tokenizer,
            DIAGNOSIS_REPAIR_TEMPLATE.format(
                question=row["question"],
                previous_response=row["initial_response"],
                feedback=row["coarse_feedback"],
                **row["diagnosis"],
            ),
        )
        for row in failed_rows
    ]

    coarse_outputs = generate_texts(llm, coarse_prompts, args.max_tokens) if coarse_prompts else []
    diagnosis_outputs = generate_texts(llm, diagnosis_prompts, args.max_tokens) if diagnosis_prompts else []

    failed_by_key = {id(row): row for row in failed_rows}
    for row, prompt, output in zip(failed_rows, coarse_prompts, coarse_outputs):
        result = run_code(output, args.execution_timeout)
        row.update(
            {
                "coarse_repair_prompt": prompt,
                "coarse_repair_response": output,
                "coarse_execution_state": result.state,
                "coarse_execution_best_solution": result.best_solution,
                "coarse_execution_stdout": result.stdout,
                "coarse_execution_stderr": result.stderr,
            }
        )
    for row, prompt, output in zip(failed_rows, diagnosis_prompts, diagnosis_outputs):
        result = run_code(output, args.execution_timeout)
        row.update(
            {
                "diagnosis_repair_prompt": prompt,
                "diagnosis_repair_response": output,
                "diagnosis_execution_state": result.state,
                "diagnosis_execution_best_solution": result.best_solution,
                "diagnosis_execution_stdout": result.stdout,
                "diagnosis_execution_stderr": result.stderr,
            }
        )

    enriched_rows = []
    failed_iter = iter(failed_rows)
    failed_map = {row["question"]: row for row in failed_rows}
    for row in rows:
        enriched_rows.append(failed_map.get(row["question"], row))

    with trajectory_file.open("w", encoding="utf-8") as fw:
        for row in enriched_rows:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")

    sft_count = 0
    with sft_file.open("w", encoding="utf-8") as fw:
        for row in failed_rows:
            diag_pred = row.get("diagnosis_execution_best_solution")
            coarse_pred = row.get("coarse_execution_best_solution")
            before_pred = row.get("initial_execution_best_solution")
            diag_correct = is_correct(diag_pred, row["answer"], args.numerical_err_tolerance)
            before_correct = is_correct(before_pred, row["answer"], args.numerical_err_tolerance)
            before_dist = numeric_distance(before_pred, row["answer"])
            diag_dist = numeric_distance(diag_pred, row["answer"])
            improved = (before_dist is None and diag_dist is not None) or (
                before_dist is not None and diag_dist is not None and diag_dist < before_dist
            )
            if diag_correct or (improved and row.get("diagnosis_execution_state") == "optimal"):
                prompt = SELF_REPAIR_SFT_PROMPT.format(
                    question=row["question"],
                    failed_output=row["initial_response"],
                    **row["diagnosis"],
                )
                completion = row["diagnosis_repair_response"]
                fw.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
                sft_count += 1

    metrics = {}
    metrics.update(summarize(rows, args.numerical_err_tolerance, "initial"))
    metrics.update(summarize(failed_rows, args.numerical_err_tolerance, "coarse"))
    metrics.update(summarize(failed_rows, args.numerical_err_tolerance, "diagnosis"))
    metrics["failed_sample_count"] = len(failed_rows)
    metrics["self_repair_sft_examples"] = sft_count
    metrics["trajectory_file"] = str(trajectory_file)
    metrics["sft_file"] = str(sft_file)
    with metrics_file.open("w", encoding="utf-8") as fw:
        json.dump(metrics, fw, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
