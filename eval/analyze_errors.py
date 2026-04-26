import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def as_float(value: Any) -> float | None:
    if value is None or value == "No Best Solution":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_correct(prediction: Any, answer: Any, tolerance: float) -> bool:
    if answer == "No Best Solution":
        return prediction == answer
    gt = as_float(answer)
    pred = as_float(prediction)
    if gt is None or pred is None:
        return False
    if gt == 0:
        return abs(pred) <= tolerance
    return abs((pred - gt) / gt) <= tolerance


def relative_error(prediction: Any, answer: Any) -> float | None:
    gt = as_float(answer)
    pred = as_float(prediction)
    if gt is None or pred is None:
        return None
    if gt == 0:
        return abs(pred)
    return abs((pred - gt) / gt)


def extract_code(text: str) -> str:
    start = text.find("```python")
    if start == -1:
        return ""
    end = text.find("```", start + len("```python"))
    if end == -1:
        return ""
    return text[start + len("```python"):end]


def infer_objective_direction(text: str) -> str | None:
    lower = text.lower()
    min_pos = min([p for p in [lower.find("minimize"), lower.find("minimise"), lower.find("minimization")] if p != -1] or [10**9])
    max_pos = min([p for p in [lower.find("maximize"), lower.find("maximise"), lower.find("maximization")] if p != -1] or [10**9])
    if min_pos == 10**9 and max_pos == 10**9:
        return None
    return "min" if min_pos < max_pos else "max"


def infer_model_sense(code: str, response: str) -> str | None:
    text = f"{response}\n{code}".lower()
    if "maximize" in text or "copt.maximize" in text or "sense=copt.maximize" in text:
        return "max"
    if "minimize" in text or "copt.minimize" in text or "sense=copt.minimize" in text:
        return "min"
    return None


def has_integrality_hint(question: str) -> bool:
    q = question.lower()
    hints = [
        "binary",
        "integer",
        "whether",
        "choose",
        "select",
        "assign",
        "number of",
        "how many",
        "at most one",
        "only one",
        "can only",
    ]
    return any(h in q for h in hints)


def has_integer_or_binary_var(code: str, response: str) -> bool:
    text = f"{code}\n{response}".lower()
    patterns = ["binary", "integer", "copt.binary", "copt.integer", "vtype=copt.b", "vtype=copt.i"]
    return any(p in text for p in patterns)


def count_constraints(code: str) -> int:
    lower = code.lower()
    return lower.count(".addconstr") + lower.count(".addconstrs") + lower.count("addconstr(") + lower.count("addconstrs(")


def suspicious_constraint_omissions(question: str, response: str, code: str) -> list[str]:
    q = question.lower()
    text = f"{response}\n{code}".lower()
    categories = {
        "capacity/budget/resource": ["capacity", "budget", "available", "limit", "at most", "no more than", "resource"],
        "demand/minimum": ["demand", "require", "required", "at least", "minimum", "satisfy"],
        "assignment/selection": ["assign", "choose", "select", "only one", "at most one", "can only"],
        "inventory/time balance": ["inventory", "storage", "month", "quarter", "period", "week", "balance"],
        "flow/supply-demand": ["supply", "origin", "destination", "ship", "transport", "flow"],
        "upper/lower bounds": ["upper", "lower", "between", "not exceed", "no more than", "at least"],
    }
    missing = []
    for name, words in categories.items():
        if any(w in q for w in words):
            if not any(w in text for w in words[: max(2, len(words) // 2)]):
                missing.append(name)
    return missing


def classify_wrong(example: dict, tolerance: float) -> tuple[str, list[str]]:
    question = example.get("question") or example.get("en_question") or ""
    response = example.get("en_math_model_coptpy_code") or ""
    code = extract_code(response)
    state = example.get("execution_state") or ""
    pred = example.get("execution_best_solution")
    answer = example.get("answer")
    tags = []

    state_lower = state.lower()
    if not code:
        return "format/no_code", ["no_code"]
    if "failed" in state_lower:
        result = (example.get("execution_result") or "").lower()
        if "syntaxerror" in result:
            return "execution/syntax_error", ["syntax_error"]
        if "nameerror" in result or "not defined" in result:
            return "execution/missing_variable", ["missing_variable"]
        if "timeout" in state_lower:
            return "execution/timeout", ["timeout"]
        return "execution/runtime_error", ["runtime_error"]
    if pred is None:
        return "execution/no_answer", ["no_answer"]
    if pred == "No Best Solution":
        text = (example.get("execution_result") or "").lower()
        if "infeasible" in text:
            tags.append("infeasible")
        elif "unbounded" in text:
            tags.append("unbounded")
        else:
            tags.append("non_optimal")
        return "solver/non_optimal", tags

    gt = as_float(answer)
    pv = as_float(pred)
    if gt is not None and pv is not None:
        q_dir = infer_objective_direction(question)
        model_dir = infer_model_sense(code, response)
        if q_dir and model_dir and q_dir != model_dir:
            tags.append("objective_direction_mismatch")
        if has_integrality_hint(question) and not has_integer_or_binary_var(code, response):
            tags.append("missing_integrality_or_binary")
        omissions = suspicious_constraint_omissions(question, response, code)
        tags.extend(f"possible_missing_{x}" for x in omissions)
        cons = count_constraints(code)
        if cons <= 1:
            tags.append("very_few_constraints")
        rel = relative_error(pred, answer)
        if rel is not None:
            if rel <= tolerance * 2:
                tags.append("near_miss")
            elif pv > gt:
                tags.append("numeric_too_high")
            elif pv < gt:
                tags.append("numeric_too_low")
        if not tags:
            tags.append("numeric_mismatch_unclear")
        return "numeric/wrong_answer", tags

    return "other/unclear", ["unclear"]


def dataset_name(path: Path) -> str:
    name = path.parent.name
    if name.startswith("eval.") and ".pass" in name:
        return name.split(".")[1]
    return name


def analyze_file(path: Path, tolerance: float) -> dict:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    total = len(rows)
    correct = [r for r in rows if is_correct(r.get("execution_best_solution"), r.get("answer"), tolerance)]
    wrong = [r for r in rows if not is_correct(r.get("execution_best_solution"), r.get("answer"), tolerance)]

    category_counts = Counter()
    tag_counts = Counter()
    examples_by_category = defaultdict(list)
    rel_errors = []
    for row in wrong:
        category, tags = classify_wrong(row, tolerance)
        category_counts[category] += 1
        tag_counts.update(tags)
        if len(examples_by_category[category]) < 3:
            examples_by_category[category].append(
                {
                    "question": (row.get("question") or "")[:240],
                    "answer": row.get("answer"),
                    "prediction": row.get("execution_best_solution"),
                    "state": row.get("execution_state"),
                    "tags": tags,
                    "relative_error": relative_error(row.get("execution_best_solution"), row.get("answer")),
                }
            )
        rel = relative_error(row.get("execution_best_solution"), row.get("answer"))
        if rel is not None and math.isfinite(rel):
            rel_errors.append(rel)

    rel_errors_sorted = sorted(rel_errors)
    median_rel = rel_errors_sorted[len(rel_errors_sorted) // 2] if rel_errors_sorted else None
    return {
        "dataset": dataset_name(path),
        "file": str(path),
        "total": total,
        "correct": len(correct),
        "wrong": len(wrong),
        "pass@1": len(correct) / total if total else 0.0,
        "wrong_category_counts": dict(category_counts),
        "wrong_tag_counts": dict(tag_counts),
        "wrong_numeric_relative_error_median": median_rel,
        "examples_by_category": dict(examples_by_category),
    }


def write_markdown(report: dict, output_path: Path):
    lines = []
    lines.append("# OR-R1 Error Diagnosis\n")
    lines.append("This report diagnoses wrong `pass@1` samples from existing executed evaluation files. It is heuristic and intended to identify repairable error classes.\n")
    lines.append("## Summary\n")
    lines.append("| Dataset | Total | Correct | Wrong | Pass@1 | Top Wrong Categories |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for item in report["datasets"]:
        cats = ", ".join(f"{k}: {v}" for k, v in Counter(item["wrong_category_counts"]).most_common(3))
        lines.append(f"| {item['dataset']} | {item['total']} | {item['correct']} | {item['wrong']} | {item['pass@1']:.4f} | {cats} |")
    lines.append("\n## Aggregate Wrong Categories\n")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for k, v in Counter(report["aggregate_wrong_category_counts"]).most_common():
        lines.append(f"| {k} | {v} |")
    lines.append("\n## Aggregate Diagnostic Tags\n")
    lines.append("| Tag | Count |")
    lines.append("|---|---:|")
    for k, v in Counter(report["aggregate_wrong_tag_counts"]).most_common(20):
        lines.append(f"| {k} | {v} |")
    lines.append("\n## Example Wrong Cases\n")
    for item in report["datasets"]:
        lines.append(f"\n### {item['dataset']}\n")
        for category, examples in item["examples_by_category"].items():
            lines.append(f"**{category}**")
            for ex in examples[:2]:
                lines.append(f"- pred=`{ex['prediction']}`, gt=`{ex['answer']}`, state=`{ex['state']}`, tags={ex['tags']}")
                lines.append(f"  question: {ex['question'].replace('|', '/')}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="eval.*.pass1/executed.jsonl")
    parser.add_argument("--numerical_err_tolerance", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {root / args.pattern}")

    datasets = [analyze_file(path, args.numerical_err_tolerance) for path in files]
    aggregate_categories = Counter()
    aggregate_tags = Counter()
    for item in datasets:
        aggregate_categories.update(item["wrong_category_counts"])
        aggregate_tags.update(item["wrong_tag_counts"])

    report = {
        "input_root": str(root),
        "num_datasets": len(datasets),
        "datasets": datasets,
        "aggregate_wrong_category_counts": dict(aggregate_categories),
        "aggregate_wrong_tag_counts": dict(aggregate_tags),
    }
    json_path = output_dir / "error_analysis.json"
    md_path = output_dir / "error_analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
