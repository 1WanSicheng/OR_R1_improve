import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from eval.analyze_errors import classify_wrong
from eval.generate_repair_lowmem import apply_chat, is_correct, numeric_distance


INITIAL_TEMPLATE_V2 = r"""
Below is an operations research question. Build a correct mathematical model and corresponding executable Python code using `coptpy`.

Requirements:
1. The final answer must follow the exact section structure below.
2. The final code must be inside a fenced ```python block.
3. The code must create `env`, define and solve `model`, and print the objective value.
4. Do not omit the code block. Do not answer with explanation only.

# Question:
{Question}

Return the answer exactly in this format:
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


FORMAT_REPAIR_TEMPLATE = r"""
You previously answered an operations research question, but the answer did not contain a valid executable Python code block.

# Question:
{question}

# Previous Response:
{previous_response}

# Feedback:
{feedback}

Task:
- Keep the modeling content only if it is still relevant.
- Rewrite the final answer into the exact required format.
- The final answer must contain a fenced ```python block with executable `coptpy` code.
- The code must define and solve `model`.

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


EXECUTION_REPAIR_TEMPLATE = r"""
You previously generated an operations research solution, but the failure is primarily a code execution issue.

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

Task:
- Fix the code/API/import/indexing issue first.
- Preserve the intended optimization model whenever possible.
- Do not change the mathematical formulation unless it is required to make the code valid.
- The final code must define and solve `model`.

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


MODELING_REPAIR_TEMPLATE = r"""
You previously generated an operations research solution, but the failure is primarily a modeling issue.

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

Task:
- Repair the formulation with targeted changes.
- Focus on variable domains, objective direction, bounds, resource constraints, and missing structural constraints.
- Keep correct parts, but fix the optimization model so the final code is executable and closer to the correct answer.
- The final code must define and solve `model`.

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


SELF_REPAIR_SFT_PROMPT_V2 = r"""
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

# Quality Target
quality_tier: {quality_tier}

Produce a corrected mathematical model and executable `coptpy` code.
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
    extraction_mode: str


def load_samples(path: str, split: str, max_samples: int | None):
    if path.endswith(("jsonl", "json")):
        ds = datasets.load_dataset("json", data_files=path)["train"]
    else:
        ds = datasets.load_dataset(path)[split]
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]
    return rows


def extract_code_v2(text: str) -> tuple[str | None, str]:
    patterns = [
        (r"```python\s*(.*?)```", "python_fence"),
        (r"```py\s*(.*?)```", "py_fence"),
        (r"```\s*(.*?)```", "generic_fence"),
    ]
    for pattern, mode in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code, mode

    heading_match = re.search(
        r"##\s*Python Code Solution Using `?coptpy`?:?\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if heading_match:
        block = heading_match.group(1).strip()
        block = re.sub(r"^```(?:python|py)?", "", block, flags=re.IGNORECASE).strip()
        block = re.sub(r"```$", "", block).strip()
        lines = block.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "env ", "env=", "model ", "model=")):
                code = "\n".join(lines[idx:]).strip()
                if code:
                    return code, "section_recovery"

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and ("copt" in stripped or "pandas" in stripped):
            code = "\n".join(lines[idx:]).strip()
            if "model" in code:
                return code, "import_recovery"
    return None, "none"


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
        if "nameerror" in text or "missing_variable" in text or "missing_model_variable" in text or "not defined" in text:
            return "missing_variable"
        return "runtime_error"
    if "unbounded" in text:
        return "unbounded"
    if "infeasible" in text:
        return "infeasible"
    if "no best solution" in text:
        return "non_optimal"
    return "unexpected_no_solution"


def run_code_v2(response: str, timeout: int) -> ExecutionResult:
    code, extraction_mode = extract_code_v2(response)
    if not code:
        return ExecutionResult("no_code", None, "", "", None, extraction_mode)

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
        return ExecutionResult(state, best, stdout, stderr, script, extraction_mode)
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult("timeout", None, exc.stdout or "", exc.stderr or "", script, extraction_mode)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def error_excerpt(stdout: str, stderr: str, max_lines: int = 10) -> str:
    text = (stderr or stdout or "").strip()
    if not text:
        return ""
    return "\n".join(text.splitlines()[-max_lines:])


def infer_route(initial_state: str) -> str:
    if initial_state == "no_code":
        return "format"
    if initial_state in {"missing_variable", "runtime_error", "syntax_error", "timeout"}:
        return "execution"
    return "modeling"


def diagnosis_from_row_v2(row: dict, tolerance: float) -> dict:
    state = row["initial_execution_state"]
    stdout = row.get("initial_execution_stdout", "")
    stderr = row.get("initial_execution_stderr", "")
    text = f"{stdout}\n{stderr}"
    tags = []
    cause_bits = []
    instruction_bits = []
    category = ""

    if state == "no_code":
        category = "format/no_code"
        tags = ["missing_fenced_code"]
        cause_bits.append("The response did not contain a recoverable executable Python code block.")
        instruction_bits.append("Return the answer in the exact required format and include a fenced `python` code block.")
    elif state == "missing_variable":
        category = "execution/missing_variable"
        lower = text.lower()
        if "pd" in lower and "not defined" in lower:
            tags.append("missing_import_pandas")
            instruction_bits.append("Add the missing pandas import and keep file-reading code consistent with the problem statement.")
        if "not defined" in lower:
            tags.append("undefined_symbol")
            instruction_bits.append("Define all referenced symbols, arrays, and decision variables before using them.")
        cause_bits.append("The code references a variable, import, or symbol that was never defined.")
    elif state == "syntax_error":
        category = "execution/syntax_error"
        tags = ["syntax_error"]
        cause_bits.append("The generated code has a Python syntax problem.")
        instruction_bits.append("Fix the syntax issue without changing unrelated parts of the model.")
    elif state == "runtime_error":
        category = "execution/runtime_error"
        lower = text.lower()
        if "'var' object is not subscriptable" in lower:
            tags.append("var_indexing_misuse")
            instruction_bits.append("Fix decision-variable container creation so indexed access matches the variable structure.")
        if "qconstrbuilder" in lower or "mqconstrbuilder" in lower:
            tags.append("invalid_copt_quadratic_api")
            instruction_bits.append("Use the correct COPT API for quadratic or second-order cone constraints.")
        if "keyerror" in lower:
            tags.append("index_key_mismatch")
            instruction_bits.append("Align loop indices and dictionary keys with how decision variables are created.")
        if not tags:
            tags.append("generic_runtime_error")
            instruction_bits.append("Fix the code/API/runtime issue while preserving the intended model.")
        cause_bits.append("The code is structurally present but crashes during execution.")
    elif state == "infeasible":
        category = "solver/infeasible"
        tags = ["infeasible"]
        cause_bits.append("The optimization model is over-constrained or has conflicting signs, bounds, or equalities.")
        instruction_bits.append("Check constraint directions, redundant equalities, lower/upper bounds, and conservation constraints for conflicts.")
    elif state == "unbounded":
        category = "solver/unbounded"
        tags = ["unbounded"]
        cause_bits.append("The model is missing bounding constraints or resource-limiting structure.")
        instruction_bits.append("Add missing non-negativity, upper bounds, capacity, budget, or linking constraints.")
    else:
        pseudo = {
            "question": row["question"],
            "answer": row["answer"],
            "en_math_model_coptpy_code": row["initial_response"],
            "execution_state": row["initial_execution_state"],
            "execution_best_solution": row["initial_execution_best_solution"],
            "execution_result": text,
        }
        category, tags = classify_wrong(pseudo, tolerance)
        tags = list(dict.fromkeys(tags))
        if "missing_integrality_or_binary" in tags:
            instruction_bits.append("Use binary or integer variables where the problem describes count, assignment, or yes/no choices.")
        if "objective_direction_mismatch" in tags:
            instruction_bits.append("Correct the objective sense and ensure the code and math model use the same optimization direction.")
        if any(t.startswith("possible_missing_") for t in tags):
            missing = [t.replace("possible_missing_", "") for t in tags if t.startswith("possible_missing_")]
            instruction_bits.append("Add the suspected missing structural constraints: " + ", ".join(missing) + ".")
        if "numeric_too_high" in tags:
            instruction_bits.append("Look for missing limiting constraints, weak upper bounds, or relaxed domains.")
        if "numeric_too_low" in tags:
            instruction_bits.append("Look for over-restrictive constraints, wrong coefficients, or overly tight bounds.")
        cause_bits.append("The code executes, but the optimization model does not match the intended problem closely enough.")

    if not instruction_bits:
        instruction_bits.append("Repair only the likely cause and keep correct parts unchanged.")
    return {
        "failure_type": category,
        "diagnostic_tags": list(dict.fromkeys(tags)),
        "likely_cause": " ".join(cause_bits),
        "repair_instruction": " ".join(instruction_bits),
        "repair_route": infer_route(state),
    }


def build_feedback_v2(row: dict, result: ExecutionResult) -> str:
    lines = [
        f"- Execution state: {result.state}",
        f"- Code extraction mode: {result.extraction_mode}",
    ]
    if result.best_solution is not None:
        lines.append(f"- Solver objective value: {result.best_solution}")
        dist = numeric_distance(result.best_solution, row["answer"])
        if dist is not None:
            lines.append(f"- Relative distance to reference answer: {dist:.6g}")
    else:
        lines.append("- No valid objective value was produced.")

    excerpt = error_excerpt(result.stdout, result.stderr)
    if excerpt:
        lines.append("- Error/solver excerpt:")
        lines.append(excerpt)

    if result.state in {"infeasible", "unbounded", "non_optimal", "unexpected_no_solution"}:
        lines.append("- The solver did not return an optimal solution. Check bounds, variable domains, objective sense, and structural constraints.")
    lines.append("- The final code must define and solve `model` and print the objective value.")
    return "\n".join(lines)


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
    executable = 0
    improved = 0
    for row in rows:
        state = row.get(f"{prefix}_execution_state")
        pred = row.get(f"{prefix}_execution_best_solution")
        if state != "no_code":
            executable += 1
        if state == "optimal":
            optimal += 1
        if is_correct(pred, row["answer"], tolerance):
            correct += 1
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
        f"{prefix}_executable_rate": executable / total if total else 0.0,
        f"{prefix}_improved_rate": improved / total if total and prefix != "initial" else 0.0,
    }


def acceptance_tier(row: dict, tolerance: float) -> str | None:
    before_state = row.get("initial_execution_state")
    after_state = row.get("diagnosis_execution_state")
    before_pred = row.get("initial_execution_best_solution")
    after_pred = row.get("diagnosis_execution_best_solution")
    before_dist = numeric_distance(before_pred, row["answer"])
    after_dist = numeric_distance(after_pred, row["answer"])

    if before_state == "no_code" and after_state != "no_code":
        return "tier1_code_recovery"
    if before_state in {"missing_variable", "runtime_error", "syntax_error", "timeout"} and after_state == "optimal":
        return "tier2_execution_fix"
    if before_state in {"infeasible", "unbounded", "non_optimal"} and after_state == "optimal":
        return "tier3_model_fix"
    if after_state == "optimal" and is_correct(after_pred, row["answer"], tolerance):
        return "tier4_correct_answer"
    if before_dist is None and after_dist is not None and after_state == "optimal":
        return "tier3_model_fix"
    if before_dist is not None and after_dist is not None and after_dist < before_dist and after_state == "optimal":
        return "tier3_numeric_improved"
    return None


def prompt_for_route(tokenizer, row: dict, kind: str) -> str:
    diagnosis = row["diagnosis"]
    if kind == "coarse":
        route = diagnosis["repair_route"]
        if route == "format":
            template = FORMAT_REPAIR_TEMPLATE
        elif route == "execution":
            template = EXECUTION_REPAIR_TEMPLATE
        else:
            template = MODELING_REPAIR_TEMPLATE
        content = template.format(
            question=row["question"],
            previous_response=row["initial_response"],
            feedback=row["coarse_feedback"],
            failure_type="generic_feedback",
            diagnostic_tags=["generic_feedback"],
            likely_cause="Use the execution evidence only.",
            repair_instruction="Repair the most likely cause visible from the feedback.",
        )
        return apply_chat(tokenizer, content)

    route = diagnosis["repair_route"]
    if route == "format":
        template = FORMAT_REPAIR_TEMPLATE
    elif route == "execution":
        template = EXECUTION_REPAIR_TEMPLATE
    else:
        template = MODELING_REPAIR_TEMPLATE
    content = template.format(
        question=row["question"],
        previous_response=row["initial_response"],
        feedback=row["coarse_feedback"],
        **diagnosis,
    )
    return apply_chat(tokenizer, content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_failed", type=int, default=48)
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
        initial_prompts.append(apply_chat(tokenizer, INITIAL_TEMPLATE_V2.replace("{Question}", question.strip())))
    initial_outputs = generate_texts(llm, initial_prompts, args.max_tokens)

    rows = []
    failed_rows = []
    for sample, prompt, output in zip(samples, initial_prompts, initial_outputs):
        question = sample.get("en_question") or sample["question"]
        result = run_code_v2(output, args.execution_timeout)
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
                "initial_extraction_mode": result.extraction_mode,
            }
        )
        if not is_correct(result.best_solution, row["answer"], args.numerical_err_tolerance):
            failed_rows.append(row)
        rows.append(row)

    failed_rows = failed_rows[: args.max_failed]
    for row in failed_rows:
        initial_result = ExecutionResult(
            state=row["initial_execution_state"],
            best_solution=row.get("initial_execution_best_solution"),
            stdout=row.get("initial_execution_stdout", ""),
            stderr=row.get("initial_execution_stderr", ""),
            script=None,
            extraction_mode=row.get("initial_extraction_mode", "unknown"),
        )
        row["coarse_feedback"] = build_feedback_v2(row, initial_result)
        row["diagnosis"] = diagnosis_from_row_v2(row, args.numerical_err_tolerance)

    coarse_prompts = [prompt_for_route(tokenizer, row, "coarse") for row in failed_rows]
    diagnosis_prompts = [prompt_for_route(tokenizer, row, "diagnosis") for row in failed_rows]

    coarse_outputs = generate_texts(llm, coarse_prompts, args.max_tokens) if coarse_prompts else []
    diagnosis_outputs = generate_texts(llm, diagnosis_prompts, args.max_tokens) if diagnosis_prompts else []

    for row, prompt, output in zip(failed_rows, coarse_prompts, coarse_outputs):
        result = run_code_v2(output, args.execution_timeout)
        row.update(
            {
                "coarse_repair_prompt": prompt,
                "coarse_repair_response": output,
                "coarse_execution_state": result.state,
                "coarse_execution_best_solution": result.best_solution,
                "coarse_execution_stdout": result.stdout,
                "coarse_execution_stderr": result.stderr,
                "coarse_extraction_mode": result.extraction_mode,
            }
        )
    for row, prompt, output in zip(failed_rows, diagnosis_prompts, diagnosis_outputs):
        result = run_code_v2(output, args.execution_timeout)
        row.update(
            {
                "diagnosis_repair_prompt": prompt,
                "diagnosis_repair_response": output,
                "diagnosis_execution_state": result.state,
                "diagnosis_execution_best_solution": result.best_solution,
                "diagnosis_execution_stdout": result.stdout,
                "diagnosis_execution_stderr": result.stderr,
                "diagnosis_extraction_mode": result.extraction_mode,
            }
        )

    failed_map = {row["question"]: row for row in failed_rows}
    enriched_rows = [failed_map.get(row["question"], row) for row in rows]

    with trajectory_file.open("w", encoding="utf-8") as fw:
        for row in enriched_rows:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")

    sft_count = 0
    tier_counts: dict[str, int] = {}
    with sft_file.open("w", encoding="utf-8") as fw:
        for row in failed_rows:
            tier = acceptance_tier(row, args.numerical_err_tolerance)
            row["accepted_quality_tier"] = tier
            if not tier:
                continue
            prompt = SELF_REPAIR_SFT_PROMPT_V2.format(
                question=row["question"],
                failed_output=row["initial_response"],
                quality_tier=tier,
                **row["diagnosis"],
            )
            completion = row["diagnosis_repair_response"]
            fw.write(json.dumps({"prompt": prompt, "completion": completion, "quality_tier": tier}, ensure_ascii=False) + "\n")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            sft_count += 1

    metrics = {}
    metrics.update(summarize(rows, args.numerical_err_tolerance, "initial"))
    metrics.update(summarize(failed_rows, args.numerical_err_tolerance, "coarse"))
    metrics.update(summarize(failed_rows, args.numerical_err_tolerance, "diagnosis"))
    metrics["failed_sample_count"] = len(failed_rows)
    metrics["self_repair_sft_examples"] = sft_count
    metrics["sft_quality_tiers"] = tier_counts
    metrics["initial_no_code_count"] = sum(r.get("initial_execution_state") == "no_code" for r in rows)
    metrics["coarse_no_code_count"] = sum(r.get("coarse_execution_state") == "no_code" for r in failed_rows)
    metrics["diagnosis_no_code_count"] = sum(r.get("diagnosis_execution_state") == "no_code" for r in failed_rows)
    metrics["initial_extraction_modes"] = {
        mode: sum(r.get("initial_extraction_mode") == mode for r in rows)
        for mode in sorted({r.get("initial_extraction_mode") for r in rows})
    }
    metrics["diagnosis_route_counts"] = {
        route: sum(r["diagnosis"]["repair_route"] == route for r in failed_rows)
        for route in sorted({r["diagnosis"]["repair_route"] for r in failed_rows})
    }
    metrics["trajectory_file"] = str(trajectory_file)
    metrics["sft_file"] = str(sft_file)

    with metrics_file.open("w", encoding="utf-8") as fw:
        json.dump(metrics, fw, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
