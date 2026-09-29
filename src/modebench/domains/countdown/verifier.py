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


"""Domain parsing, execution, and canonical identities."""
from __future__ import annotations
import ast
import re
from collections import Counter
from fractions import Fraction
from typing import Any

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



def validate_reference(spec, level):
    """Validate this domain reference before any response is scored."""
    from modebench.validation import InputError, positive_integer
    numbers = spec.get('numbers')
    if not isinstance(numbers, list) or not numbers:
        raise InputError('answer.numbers must be a nonempty list of integers')
    for label, value in [('target', spec.get('target'))] + [('numbers', n) for n in numbers]:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InputError(f'answer.{label} must contain integers')
