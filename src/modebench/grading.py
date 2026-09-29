# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Executable validation and canonical identities for the five ModeBench domains."""
from __future__ import annotations
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from .mathir import MATHIR_MENU_VERIFIER, MATHIR_VERIFIER, validate_mathir_action_menu, validate_mathir_algebra
from .pantry_plan import PANTRY_PLAN_VERIFIER, validate_pantry_plan
from .python_modebench import PYTHON_FACTOR_VERIFIER, python_factor_route_signature
from .python_modebench_process import validate_python_factor_function_external

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None

def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

def _parse_modebench_spec(gt_answer: Any) -> dict[str, Any] | None:
    if isinstance(gt_answer, dict):
        spec = gt_answer
    elif isinstance(gt_answer, str):
        text = gt_answer.strip()
        if not (text.startswith("{") and text.endswith("}")):
            return None
        try:
            spec = json.loads(text)
        except Exception:
            return None
    else:
        return None
    verifier = spec.get("verifier")
    if verifier in {
        "graph_coloring",
        "countdown",
        MATHIR_VERIFIER,
        MATHIR_MENU_VERIFIER,
        PANTRY_PLAN_VERIFIER,
        PYTHON_FACTOR_VERIFIER,
    }:
        return spec
    return None

PYTHON_FACTOR_RESPONSE_SURFACE_VERSION = (
    "python-factor-response-v2-latex-lambda"
)

def _normalize_python_factor_lambda_surface(candidate: str) -> str:
    """Normalize one formatting-only LaTeX spelling of ``lambda n:``."""

    # Qwen's native boxed surface can typeset the Python keyword as the LaTeX
    # command ``\lambda``.  Accept only the exact required signature (plus the
    # conventional ``\,`` spacing alias); the restricted AST parser and
    # isolated executable validator remain the authority.
    return re.sub(
        r"^\\lambda(?:\s+|\\,\s*)n\s*:\s*",
        "lambda n: ",
        str(candidate).strip(),
        count=1,
    ).strip()

def _extract_modebench_candidate(model_response: str, gt_answer: Any) -> str | None:
    spec = _parse_modebench_spec(gt_answer)
    if spec is None:
        return None
    if "\\boxed" in str(model_response):
        candidate = extract_answer(str(model_response))
        if candidate is None:
            return None
        if str(spec.get("verifier")) == PYTHON_FACTOR_VERIFIER:
            candidate = _normalize_python_factor_lambda_surface(candidate)
        return candidate or None
    candidate = str(model_response).strip()
    if not candidate:
        return None
    if str(spec.get("verifier")) in {
        MATHIR_VERIFIER,
        MATHIR_MENU_VERIFIER,
    }:
        fenced = re.fullmatch(
            r"```(?:mathir)?\s*(.*?)\s*```",
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced is not None:
            candidate = fenced.group(1).strip()
        if len(candidate) > 200 or "<" in candidate or ">" in candidate:
            return None
        # Whitespace, a terminal semicolon, and an all-program code fence are
        # formatting aliases.  The restricted parser remains the authority.
        return candidate or None
    if str(spec.get("verifier")) == PYTHON_FACTOR_VERIFIER:
        fenced = re.fullmatch(
            r"```(?:python)?\s*(.*?)\s*```",
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced is not None:
            candidate = fenced.group(1).strip()
        if (
            len(candidate) > 240
            or "\n" in candidate
            or "\r" in candidate
            or "<" in candidate
            or ">" in candidate
        ):
            return None
        candidate = re.sub(
            r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|program|function)\s*"
            r"(?:is|=|:)\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = _normalize_python_factor_lambda_surface(candidate)
        return candidate or None
    if str(spec.get("verifier")) == PANTRY_PLAN_VERIFIER:
        if (
            len(candidate) > 512
            or "\n" in candidate
            or "\r" in candidate
            or "<" in candidate
            or ">" in candidate
        ):
            return None
        candidate = re.sub(
            r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|plan|recipe)\s*"
            r"(?:is|=|:)\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        return candidate or None
    if "\n" in candidate or "\r" in candidate or "<" in candidate or ">" in candidate:
        return None
    if len(candidate) > 160:
        return None
    candidate = re.sub(
        r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|expression|coloring|program)\s*"
        r"(?:is|=|:)\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    return candidate or None

def _parse_graph_coloring_answer(candidate: str, n: int) -> list[int] | None:
    text = str(candidate).strip().lower()
    text = re.sub(r"^(?:coloring|answer)\s*(?:is|=|:)\s*", "", text).strip()
    compact = re.sub(r"[\s,;|\[\]\(\)\{\}:.-]+", "", text)
    if len(compact) == n and all(ch in "123" for ch in compact):
        return [int(ch) for ch in compact]
    tokens = re.findall(r"[123]", text)
    if len(tokens) == n:
        return [int(token) for token in tokens]
    return None

def _parse_graph_digit_sequence(candidate: str, expected_len: int) -> list[int] | None:
    text = str(candidate).strip().lower()
    text = re.sub(
        r"^(?:missing\s+)?(?:colors?|digits?|answer)\s*(?:are|is|=|:)\s*",
        "",
        text,
    ).strip()
    compact = re.sub(r"[\s,;|\[\]\(\)\{\}:.-]+", "", text)
    if len(compact) == expected_len and all(ch in "123" for ch in compact):
        return [int(ch) for ch in compact]
    tokens = re.findall(r"[123]", text)
    if len(tokens) == expected_len:
        return [int(token) for token in tokens]
    return None

def _verify_graph_coloring_colors(
    colors: list[int] | None,
    spec: dict[str, Any],
) -> bool:
    """Validate one already-parsed coloring object against its problem."""

    try:
        n = int(spec["n"])
        edges = spec["edges"]
    except Exception:
        return False
    if colors is None or len(colors) != n:
        return False
    partial_colors = spec.get("partial_colors")
    if partial_colors is not None:
        try:
            if len(partial_colors) != n:
                return False
            for index, color in enumerate(partial_colors):
                if color is None:
                    continue
                if colors[index] != int(color):
                    return False
        except Exception:
            return False
    for edge in edges:
        try:
            u, v = int(edge[0]), int(edge[1])
        except Exception:
            return False
        if u < 1 or v < 1 or u > n or v > n:
            return False
        if colors[u - 1] == colors[v - 1]:
            return False
    return True

def _verify_graph_coloring_answer(candidate: str, spec: dict[str, Any]) -> bool:
    colors = _graph_coloring_from_candidate(candidate, spec)
    return _verify_graph_coloring_colors(colors, spec)

def _normalize_countdown_expression(candidate: str) -> str:
    text = str(candidate).strip()
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("\\div", "/").replace("÷", "/").replace("×", "*")
    text = text.replace("^", "**")
    text = re.sub(r"^\s*(?:expression|answer)\s*(?:is|=|:)\s*", "", text, flags=re.I)
    return text.strip()

def _countdown_eval_and_numbers(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Expression):
        return _countdown_eval_and_numbers(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("Countdown constants must be integers.")
        return Fraction(int(node.value), 1), [int(node.value)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value, numbers = _countdown_eval_and_numbers(node.operand)
        if isinstance(node.op, ast.USub):
            value = -value
        return value, numbers
    if isinstance(node, ast.BinOp):
        left, left_numbers = _countdown_eval_and_numbers(node.left)
        right, right_numbers = _countdown_eval_and_numbers(node.right)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Countdown division by zero.")
            value = left / right
        else:
            raise ValueError("Unsupported Countdown operator.")
        return value, left_numbers + right_numbers
    raise ValueError("Unsupported Countdown expression.")

def _verify_countdown_expression(candidate: str, spec: dict[str, Any]) -> bool:
    try:
        target = Fraction(int(spec["target"]), 1)
        expected_numbers = Counter(int(value) for value in spec["numbers"])
    except Exception:
        return False
    text = _normalize_countdown_expression(candidate)
    if not text:
        return False
    parts = [part.strip() for part in text.split("=") if part.strip()]
    if not parts:
        parts = [text]
    for part in parts:
        if not re.fullmatch(r"[0-9+\-*/().\s*]+", part):
            continue
        try:
            parsed = ast.parse(part, mode="eval")
            value, used_numbers = _countdown_eval_and_numbers(parsed)
        except Exception:
            continue
        if value == target and Counter(used_numbers) == expected_numbers:
            return True
    return False

def _graph_coloring_from_candidate(
    candidate: str,
    spec: dict[str, Any],
) -> list[int] | None:
    try:
        n = int(spec["n"])
    except Exception:
        return None
    colors = _parse_graph_coloring_answer(candidate, n)
    partial_colors = spec.get("partial_colors")
    if colors is None and partial_colors is not None:
        try:
            hidden_positions = [
                index for index, color in enumerate(partial_colors) if color is None
            ]
            fill = _parse_graph_digit_sequence(candidate, len(hidden_positions))
            if fill is not None:
                colors = [
                    int(color) if color is not None else 0 for color in partial_colors
                ]
                for index, color in zip(hidden_positions, fill):
                    colors[index] = color
        except Exception:
            return None
    if colors is None or len(colors) != n:
        return None
    return colors

def _canonical_countdown_ast(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _canonical_countdown_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("Countdown constants must be integers.")
        return str(int(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _canonical_countdown_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return f"neg({inner})"
        return inner
    if isinstance(node, ast.BinOp):
        left = _canonical_countdown_ast(node.left)
        right = _canonical_countdown_ast(node.right)
        if isinstance(node.op, ast.Add):
            parts = sorted([left, right])
            return f"add({parts[0]},{parts[1]})"
        if isinstance(node.op, ast.Mult):
            parts = sorted([left, right])
            return f"mul({parts[0]},{parts[1]})"
        if isinstance(node.op, ast.Sub):
            return f"sub({left},{right})"
        if isinstance(node.op, ast.Div):
            return f"div({left},{right})"
    raise ValueError("Unsupported Countdown expression.")

def _canonical_countdown_route_ast(node: ast.AST) -> str:
    """Canonical operator/dependency skeleton with numeric leaves abstracted."""

    if isinstance(node, ast.Expression):
        return _canonical_countdown_route_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("Countdown constants must be integers.")
        return "input"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _canonical_countdown_route_ast(node.operand)
        return f"neg({inner})" if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left = _canonical_countdown_route_ast(node.left)
        right = _canonical_countdown_route_ast(node.right)
        if isinstance(node.op, ast.Add):
            parts = sorted([left, right])
            return f"add({parts[0]},{parts[1]})"
        if isinstance(node.op, ast.Mult):
            parts = sorted([left, right])
            return f"mul({parts[0]},{parts[1]})"
        if isinstance(node.op, ast.Sub):
            return f"sub({left},{right})"
        if isinstance(node.op, ast.Div):
            return f"div({left},{right})"
    raise ValueError("Unsupported Countdown route expression.")

def _canonical_countdown_expression_key(
    candidate: str,
    spec: dict[str, Any],
) -> str | None:
    try:
        expected_numbers = Counter(int(value) for value in spec["numbers"])
    except Exception:
        return None
    text = _normalize_countdown_expression(candidate)
    if not text:
        return None
    parts = [part.strip() for part in text.split("=") if part.strip()] or [text]
    for part in parts:
        if not re.fullmatch(r"[0-9+\-*/().\s*]+", part):
            continue
        try:
            parsed = ast.parse(part, mode="eval")
            _, used_numbers = _countdown_eval_and_numbers(parsed)
            if Counter(used_numbers) != expected_numbers:
                continue
            return f"countdown:{_canonical_countdown_ast(parsed)}"
        except Exception:
            continue
    return None

def _modebench_answer_key(
    model_response: str,
    gt_answer: Any,
) -> str | None:
    spec = _parse_modebench_spec(gt_answer)
    if spec is None:
        return None
    candidate = _extract_modebench_candidate(model_response, gt_answer)
    if candidate is None:
        return None
    verifier = str(spec.get("verifier"))
    if verifier == "graph_coloring":
        colors = _graph_coloring_from_candidate(candidate, spec)
        if colors is None:
            return None
        return "graph_coloring:" + "".join(str(int(color)) for color in colors)
    if verifier == "countdown":
        return _canonical_countdown_expression_key(candidate, spec)
    if verifier == MATHIR_VERIFIER:
        validation = validate_mathir_algebra(candidate, spec)
        return validation.canonical_key if validation is not None else None
    if verifier == MATHIR_MENU_VERIFIER:
        validation = validate_mathir_action_menu(candidate, spec)
        return validation.canonical_key if validation is not None else None
    if verifier == PYTHON_FACTOR_VERIFIER:
        validation = validate_python_factor_function_external(candidate, spec)
        return validation.canonical_key if validation is not None else None
    if verifier == PANTRY_PLAN_VERIFIER:
        validation = validate_pantry_plan(candidate, spec)
        return validation.canonical_key if validation is not None else None
    return None

def validated_modebench_outcome_key(
    model_response: str,
    gt_answer: Any,
) -> str | None:
    """Return a canonical outcome only when that same response verifies.

    This is the admission boundary for online canonical banks.  It deliberately
    supports only ModeBench tasks with executable validators:

    - graph colorings are parsed and checked against every graph edge;
    - Countdown expressions are parsed, checked for exact operand use, and
      executed against the requested target.

    - MathIR algebra programs are parsed by a restricted grammar, executed as
      state transformations, exact-normalized, and accepted only when execution
      isolates the right solution.  Their key comes from those executed states.

    - Python factor functions are syntax-restricted and called in an isolated
      Python worker.  Their key is the vector returned by those exact tool calls.

    Ordinary MATH final-answer grading is not a proof/strategy verifier and is
    therefore rejected here rather than being mislabeled as canonical search.
    """

    spec = _parse_modebench_spec(gt_answer)
    if spec is None:
        return None
    candidate = _extract_modebench_candidate(model_response, gt_answer)
    if candidate is None:
        return None
    verifier = str(spec.get("verifier"))
    if verifier == "graph_coloring":
        colors = _graph_coloring_from_candidate(candidate, spec)
        if not _verify_graph_coloring_colors(colors, spec):
            return None
        assert colors is not None
        return "graph_coloring:" + "".join(str(int(color)) for color in colors)
    if verifier == MATHIR_VERIFIER:
        validation = validate_mathir_algebra(candidate, spec)
        # The key is constructed from the exact normalized equation states
        # produced by the same interpreter execution that validated the answer.
        return validation.canonical_key if validation is not None else None
    if verifier == MATHIR_MENU_VERIFIER:
        validation = validate_mathir_action_menu(candidate, spec)
        # Menu labels are expanded first. Identity comes only from the exact
        # normalized states produced by executing those concrete operations.
        return validation.canonical_key if validation is not None else None
    if verifier == PYTHON_FACTOR_VERIFIER:
        validation = validate_python_factor_function_external(candidate, spec)
        return validation.canonical_key if validation is not None else None
    if verifier == PANTRY_PLAN_VERIFIER:
        validation = validate_pantry_plan(candidate, spec)
        return validation.canonical_key if validation is not None else None
    if verifier != "countdown":
        return None
    try:
        target = Fraction(int(spec["target"]), 1)
        expected_numbers = Counter(int(value) for value in spec["numbers"])
    except Exception:
        return None
    text = _normalize_countdown_expression(candidate)
    parts = [part.strip() for part in text.split("=") if part.strip()] or [text]
    for part in parts:
        if not re.fullmatch(r"[0-9+\-*/().\s*]+", part):
            continue
        try:
            parsed = ast.parse(part, mode="eval")
            value, used_numbers = _countdown_eval_and_numbers(parsed)
            if value != target or Counter(used_numbers) != expected_numbers:
                continue
            # The key is derived from the exact AST object that passed
            # execution and operand validation above.
            return f"countdown:{_canonical_countdown_ast(parsed)}"
        except Exception:
            continue
    return None

@dataclass(frozen=True)
class VerifiedExplorationIdentity:
    """Separate prompt-local endpoint and cross-prompt route identities."""

    verifier: str
    endpoint_key: str
    route_signature: str | None

def validated_modebench_exploration_identity(
    model_response: str,
    gt_answer: Any,
) -> VerifiedExplorationIdentity | None:
    """Return hierarchical identity only after exact executable validation."""

    spec = _parse_modebench_spec(gt_answer)
    if spec is None:
        return None
    candidate = _extract_modebench_candidate(model_response, gt_answer)
    if candidate is None:
        return None
    verifier = str(spec.get("verifier"))
    if verifier == "graph_coloring":
        colors = _graph_coloring_from_candidate(candidate, spec)
        if not _verify_graph_coloring_colors(colors, spec):
            return None
        assert colors is not None
        endpoint = "graph_coloring:" + "".join(str(int(color)) for color in colors)
        return VerifiedExplorationIdentity(verifier, endpoint, None)
    if verifier == MATHIR_VERIFIER:
        validation = validate_mathir_algebra(candidate, spec)
        if validation is None:
            return None
        solution = validation.solution
        endpoint = f"mathir-solution:{solution.numerator}/{solution.denominator}"
        return VerifiedExplorationIdentity(
            verifier,
            endpoint,
            validation.route_signature,
        )
    if verifier == MATHIR_MENU_VERIFIER:
        validation = validate_mathir_action_menu(candidate, spec)
        if validation is None:
            return None
        solution = validation.solution
        endpoint = f"mathir-solution:{solution.numerator}/{solution.denominator}"
        return VerifiedExplorationIdentity(
            verifier,
            endpoint,
            validation.route_signature,
        )
    if verifier == PYTHON_FACTOR_VERIFIER:
        validation = validate_python_factor_function_external(candidate, spec)
        if validation is None:
            return None
        try:
            route = python_factor_route_signature(candidate)
        except Exception:
            return None
        return VerifiedExplorationIdentity(
            verifier,
            validation.canonical_key,
            route,
        )
    if verifier == PANTRY_PLAN_VERIFIER:
        validation = validate_pantry_plan(candidate, spec)
        if validation is None:
            return None
        return VerifiedExplorationIdentity(
            verifier,
            validation.canonical_key,
            None,
        )
    if verifier != "countdown":
        return None
    try:
        target = Fraction(int(spec["target"]), 1)
        expected_numbers = Counter(int(value) for value in spec["numbers"])
    except Exception:
        return None
    text = _normalize_countdown_expression(candidate)
    parts = [part.strip() for part in text.split("=") if part.strip()] or [text]
    for part in parts:
        if not re.fullmatch(r"[0-9+\-*/().\s*]+", part):
            continue
        try:
            parsed = ast.parse(part, mode="eval")
            value, used_numbers = _countdown_eval_and_numbers(parsed)
            if value != target or Counter(used_numbers) != expected_numbers:
                continue
            endpoint = f"countdown-value:{value.numerator}/{value.denominator}"
            route = "countdown-route:v1:" + _canonical_countdown_route_ast(parsed)
            return VerifiedExplorationIdentity(verifier, endpoint, route)
        except Exception:
            continue
    return None

def _grade_modebench_answer(model_answer: str, gt_answer: Any) -> bool | None:
    spec = _parse_modebench_spec(gt_answer)
    if spec is None:
        return None
    verifier = str(spec.get("verifier"))
    if verifier == "graph_coloring":
        return _verify_graph_coloring_answer(model_answer, spec)
    if verifier == "countdown":
        return _verify_countdown_expression(model_answer, spec)
    if verifier == MATHIR_VERIFIER:
        return validate_mathir_algebra(model_answer, spec) is not None
    if verifier == MATHIR_MENU_VERIFIER:
        return validate_mathir_action_menu(model_answer, spec) is not None
    if verifier == PYTHON_FACTOR_VERIFIER:
        return validate_python_factor_function_external(model_answer, spec) is not None
    if verifier == PANTRY_PLAN_VERIFIER:
        return validate_pantry_plan(model_answer, spec) is not None
    return False

def boxed_reward_fn(model_response, gt_answer, fast=False):
    """Benchmark-only reward adapter; non-benchmark references fail closed."""
    candidate = _extract_modebench_candidate(model_response, gt_answer)
    if candidate is None:
        return {"formatted": False}, 0.0
    correct = _grade_modebench_answer(candidate, gt_answer)
    return {"formatted": True}, float(bool(correct))


def extract_normalized_final_answer(model_response, *, template="qwen_boxed", gt_answer=None):
    """Legacy display key. Use validated_modebench_outcome_key for admission."""
    return _modebench_answer_key(model_response, gt_answer)
