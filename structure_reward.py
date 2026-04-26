import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


OBJECTIVE_PATTERNS = [
    (re.compile(r"\bminimi[sz]e\b|\bminimum\b", re.IGNORECASE), "min"),
    (re.compile(r"\bmaximi[sz]e\b|\bmaximum\b", re.IGNORECASE), "max"),
]

VAR_TYPE_HINTS = {
    "binary": [
        "binary",
        "0-1",
        "0 or 1",
        "boolean",
        "whether to",
        "decide whether",
        "is binary",
    ],
    "integer": [
        "integer",
        "whole number",
        "whole numbers",
        "integral",
        "must be whole",
        "indivisible",
    ],
    "continuous": [
        "continuous",
        "fractional",
        "real-valued",
    ],
}

CONSTRAINT_KEYWORDS = {
    "capacity": ["capacity", "capacities", "cannot exceed", "at most", "upper bound", "limit"],
    "demand_satisfaction": ["demand", "meet demand", "satisfy demand", "requires", "need to transport"],
    "supply_balance": ["supply", "available", "availability", "surplus", "net demand", "flow balance"],
    "assignment_uniqueness": ["assign", "assignment", "exactly one", "at most one", "one project", "one route"],
    "budget": ["budget", "budgetary", "cost limit", "spend", "expenditure"],
    "inventory_conservation": ["inventory", "conservation", "holding", "stock", "carry-over"],
    "flow_balance": ["flow balance", "incoming", "outgoing", "net demand", "conservation of flow"],
    "resource_limit": ["resource", "resources", "workforce", "hours", "time available"],
    "time_consistency": ["time", "schedule", "pickup", "drop-off", "earliest", "latest", "sequence", "continuous driving"],
    "nonnegativity": ["nonnegative", "non-negativity", "greater than or equal to 0", ">= 0"],
    "integrality": ["integer", "binary", "whole number", "whole numbers"],
    "cone": ["second-order cone", "cone constraint", "soc constraint"],
}

ALIGNMENT_PATTERNS = [
    (re.compile(r"\bcosts?\b.*\bper unit\b", re.IGNORECASE), "cost_to_flow"),
    (re.compile(r"\bprice\b|\brevenue\b|\bprofit\b", re.IGNORECASE), "objective_parameter"),
    (re.compile(r"\bsupply\b|\bavailable\b|\bavailability\b", re.IGNORECASE), "supply_entity"),
    (re.compile(r"\bdemand\b|\brequires\b|\brequired\b", re.IGNORECASE), "demand_entity"),
    (re.compile(r"\bcapacity\b|\bmaximum capacity\b|\bupper bound\b", re.IGNORECASE), "capacity_entity"),
]


@dataclass
class ProblemSchema:
    objective: Optional[str]
    variable_type: Optional[str]
    variable_arity: Optional[int]
    constraints: Set[str]
    alignments: Set[str]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def infer_objective_direction(text: str) -> Optional[str]:
    for pattern, direction in OBJECTIVE_PATTERNS:
        if pattern.search(text):
            return direction
    return None


def infer_variable_type(text: str) -> Optional[str]:
    normalized = _normalize_text(text)
    for var_type, hints in VAR_TYPE_HINTS.items():
        if any(hint in normalized for hint in hints):
            return var_type
    # Default to continuous for most OR tasks if no discrete hint appears.
    if "decision variable" in normalized or "allocate" in normalized or "amount" in normalized:
        return "continuous"
    return None


def _extract_indices_from_text(text: str) -> List[Tuple[str, ...]]:
    index_matches: List[Tuple[str, ...]] = []
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\[([^\]]+)\]", text):
        indices = tuple(part.strip().lower() for part in match.group(2).split(",") if part.strip())
        if indices:
            index_matches.append(indices)
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*_\{([^}]+)\}", text):
        indices = tuple(part.strip().lower() for part in match.group(2).split(",") if part.strip())
        if indices:
            index_matches.append(indices)
    return index_matches


def infer_variable_arity(problem_text: str, completion_text: Optional[str] = None) -> Optional[int]:
    source = completion_text if completion_text is not None else problem_text
    matches = _extract_indices_from_text(source)
    if matches:
        return max(len(indices) for indices in matches)

    normalized = _normalize_text(problem_text)
    if "between each pair" in normalized or "from each" in normalized and "to each" in normalized:
        return 2
    if "for each" in normalized:
        return 1
    return None


def infer_constraints(text: str) -> Set[str]:
    normalized = _normalize_text(text)
    constraints: Set[str] = set()
    for name, hints in CONSTRAINT_KEYWORDS.items():
        if any(hint in normalized for hint in hints):
            constraints.add(name)
    return constraints


def infer_alignments(text: str) -> Set[str]:
    alignments: Set[str] = set()
    for pattern, relation in ALIGNMENT_PATTERNS:
        if pattern.search(text):
            alignments.add(relation)
    return alignments


def build_problem_schema(problem_text: str) -> ProblemSchema:
    return ProblemSchema(
        objective=infer_objective_direction(problem_text),
        variable_type=infer_variable_type(problem_text),
        variable_arity=infer_variable_arity(problem_text),
        constraints=infer_constraints(problem_text),
        alignments=infer_alignments(problem_text),
    )


def build_completion_schema(problem_text: str, completion_text: str) -> ProblemSchema:
    return ProblemSchema(
        objective=infer_objective_direction(completion_text),
        variable_type=infer_variable_type(completion_text),
        variable_arity=infer_variable_arity(problem_text, completion_text),
        constraints=infer_constraints(completion_text),
        alignments=infer_alignments(completion_text),
    )


def score_objective(problem_schema: ProblemSchema, completion_schema: ProblemSchema) -> float:
    if completion_schema.objective is None:
        return 0.0
    if problem_schema.objective is None:
        return 0.5
    return 1.0 if problem_schema.objective == completion_schema.objective else 0.3


def score_variable(problem_schema: ProblemSchema, completion_schema: ProblemSchema) -> float:
    scores: List[float] = []

    if problem_schema.variable_type is None:
        scores.append(0.5 if completion_schema.variable_type is not None else 0.0)
    elif completion_schema.variable_type == problem_schema.variable_type:
        scores.append(1.0)
    elif completion_schema.variable_type is None:
        scores.append(0.0)
    else:
        scores.append(0.5 if {problem_schema.variable_type, completion_schema.variable_type} == {"integer", "binary"} else 0.0)

    if problem_schema.variable_arity is None:
        scores.append(0.5 if completion_schema.variable_arity is not None else 0.0)
    elif completion_schema.variable_arity == problem_schema.variable_arity:
        scores.append(1.0)
    elif completion_schema.variable_arity is None:
        scores.append(0.0)
    else:
        scores.append(0.5 if abs(problem_schema.variable_arity - completion_schema.variable_arity) == 1 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def score_constraint(problem_schema: ProblemSchema, completion_schema: ProblemSchema) -> float:
    if not problem_schema.constraints:
        return 0.0
    matched = problem_schema.constraints & completion_schema.constraints
    return len(matched) / len(problem_schema.constraints)


def score_alignment(problem_schema: ProblemSchema, completion_schema: ProblemSchema) -> float:
    if not problem_schema.alignments:
        return 0.0
    matched = problem_schema.alignments & completion_schema.alignments
    return len(matched) / len(problem_schema.alignments)


def score_structure(
    problem_text: str,
    completion_text: str,
    alpha_obj: float = 1.0,
    alpha_var: float = 1.0,
    alpha_con: float = 2.0,
    alpha_align: float = 1.0,
) -> Dict[str, float]:
    problem_schema = build_problem_schema(problem_text)
    completion_schema = build_completion_schema(problem_text, completion_text)

    r_obj = score_objective(problem_schema, completion_schema)
    r_var = score_variable(problem_schema, completion_schema)
    r_con = score_constraint(problem_schema, completion_schema)
    r_align = score_alignment(problem_schema, completion_schema)

    denom = alpha_obj + alpha_var + alpha_con + alpha_align
    r_struct = (
        alpha_obj * r_obj
        + alpha_var * r_var
        + alpha_con * r_con
        + alpha_align * r_align
    ) / denom

    return {
        "r_obj": r_obj,
        "r_var": r_var,
        "r_con": r_con,
        "r_align": r_align,
        "r_struct": r_struct,
    }
