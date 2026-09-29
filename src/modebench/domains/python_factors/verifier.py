"""Restricted executable Python tasks for ModeBench.

The model writes one pure ``lambda n: ...`` expression.  A separate worker
process evaluates the function on a frozen prompt-local test suite.  A response
is correct when every returned integer is a proper divisor of its input.  The
semantic outcome is the complete vector returned by that same execution.

This module owns the syntax and task contract.  The actual call to model code
is made only by :mod:`modebench.python_modebench_worker`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import prod
from typing import Any, Mapping


PYTHON_FACTOR_VERIFIER = "python_factor_function"
PYTHON_FACTOR_VERSION = "factor-v1"
_MAX_CANDIDATE_CHARS = 240
_MAX_AST_NODES = 64
_MAX_CASES = 8

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.Lambda,
    ast.arguments,
    ast.arg,
    ast.IfExp,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Mod,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Name,
    ast.Load,
    ast.Constant,
)


class PythonModeBenchError(ValueError):
    """Raised for a malformed task or candidate program."""


@dataclass(frozen=True)
class PythonFactorValidation:
    """Result of one successful external execution."""

    canonical_key: str
    outputs: tuple[int, ...]


def _canonical_python_route_ast(node: ast.AST) -> str:
    """Return a literal-abstracted strategy skeleton for restricted Python."""

    if isinstance(node, ast.Expression):
        return _canonical_python_route_ast(node.body)
    if isinstance(node, ast.Lambda):
        return f"lambda({_canonical_python_route_ast(node.body)})"
    if isinstance(node, ast.Name):
        return "input"
    if isinstance(node, ast.Constant):
        return "literal"
    if isinstance(node, ast.UnaryOp):
        return (
            f"{type(node.op).__name__.lower()}"
            f"({_canonical_python_route_ast(node.operand)})"
        )
    if isinstance(node, ast.BinOp):
        op = type(node.op).__name__.lower()
        children = [
            _canonical_python_route_ast(node.left),
            _canonical_python_route_ast(node.right),
        ]
        if isinstance(node.op, (ast.Add, ast.Mult)):
            children.sort()
        return f"{op}({','.join(children)})"
    if isinstance(node, ast.Compare):
        return "compare(" + ",".join(
            [
                _canonical_python_route_ast(node.left),
                *(type(op).__name__.lower() for op in node.ops),
                *(
                    _canonical_python_route_ast(comparator)
                    for comparator in node.comparators
                ),
            ]
        ) + ")"
    if isinstance(node, ast.BoolOp):
        values = [_canonical_python_route_ast(value) for value in node.values]
        values.sort()
        return f"{type(node.op).__name__.lower()}({','.join(values)})"
    if isinstance(node, ast.IfExp):
        return (
            f"if({_canonical_python_route_ast(node.test)},"
            f"{_canonical_python_route_ast(node.body)},"
            f"{_canonical_python_route_ast(node.orelse)})"
        )
    raise PythonModeBenchError(
        f"cannot canonicalize Python route node {type(node).__name__}"
    )


def python_factor_route_signature(candidate: str) -> str:
    """Return code-structure identity after the restricted parser accepts it."""

    parsed = parse_python_factor_candidate(candidate)
    return "python-factor-route:v1:" + _canonical_python_route_ast(parsed)


def proper_divisors(value: int) -> tuple[int, ...]:
    """Return all positive proper divisors other than one."""

    value = int(value)
    if value < 4:
        return ()
    return tuple(divisor for divisor in range(2, value) if value % divisor == 0)


def python_factor_mode_count(cases: tuple[int, ...] | list[int]) -> int:
    """Return the exact number of valid behavior vectors for a task."""

    divisor_sets = [proper_divisors(value) for value in cases]
    if any(not divisors for divisors in divisor_sets):
        return 0
    return prod(len(divisors) for divisors in divisor_sets)


def parse_python_factor_spec(spec: Mapping[str, Any]) -> tuple[int, ...]:
    """Validate a trusted dataset specification and return its tool inputs."""

    if spec.get("verifier") != PYTHON_FACTOR_VERIFIER:
        raise PythonModeBenchError("wrong Python ModeBench verifier")
    if spec.get("python_version") != PYTHON_FACTOR_VERSION:
        raise PythonModeBenchError("unsupported Python ModeBench version")
    raw_cases = spec.get("cases")
    if not isinstance(raw_cases, list) or not 2 <= len(raw_cases) <= _MAX_CASES:
        raise PythonModeBenchError("cases must contain between two and eight inputs")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_cases):
        raise PythonModeBenchError("case inputs must be integers")
    cases = tuple(int(value) for value in raw_cases)
    if len(set(cases)) != len(cases):
        raise PythonModeBenchError("case inputs must be unique")
    if any(value < 4 or value > 1_000 for value in cases):
        raise PythonModeBenchError("case inputs are outside the bounded domain")
    if python_factor_mode_count(cases) < 2:
        raise PythonModeBenchError("task does not have multiple valid modes")
    return cases


def parse_python_factor_candidate(candidate: str) -> ast.Expression:
    """Parse the bounded, call-free lambda language."""

    text = str(candidate).strip()
    if not text or len(text) > _MAX_CANDIDATE_CHARS:
        raise PythonModeBenchError("candidate is empty or too long")
    try:
        parsed = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise PythonModeBenchError("candidate is not one Python expression") from error
    if not isinstance(parsed.body, ast.Lambda):
        raise PythonModeBenchError("candidate must be a lambda")
    arguments = parsed.body.args
    if (
        len(arguments.args) != 1
        or arguments.args[0].arg != "n"
        or arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.defaults
        or arguments.kw_defaults
    ):
        raise PythonModeBenchError("lambda must have the exact signature lambda n")
    nodes = list(ast.walk(parsed))
    if len(nodes) > _MAX_AST_NODES:
        raise PythonModeBenchError("candidate AST is too large")
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise PythonModeBenchError(
                f"unsupported Python syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id != "n":
            raise PythonModeBenchError(f"unknown name: {node.id}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise PythonModeBenchError("only integer literals are allowed")
            if abs(int(node.value)) > 10_000:
                raise PythonModeBenchError("integer literal is too large")
    return parsed


def execute_python_factor_candidate(
    candidate: str,
    spec: Mapping[str, Any],
) -> PythonFactorValidation:
    """Execute one already restricted candidate.

    This entry point is worker-only.  Callers in the trainer and evaluator use
    ``validate_python_factor_function_external`` so model code never executes
    in their process.
    """

    cases = parse_python_factor_spec(spec)
    parsed = parse_python_factor_candidate(candidate)
    code = compile(parsed, "<modebench-python-factor>", "eval")
    function = eval(code, {"__builtins__": {}}, {})  # noqa: S307
    outputs: list[int] = []
    for value in cases:
        output = function(value)
        if isinstance(output, bool) or not isinstance(output, int):
            raise PythonModeBenchError("function output must be an integer")
        output = int(output)
        if output <= 1 or output >= value or value % output != 0:
            raise PythonModeBenchError(
                f"function returned {output}, not a proper divisor of {value}"
            )
        outputs.append(output)
    output_tuple = tuple(outputs)
    key = "python_factor:" + ",".join(str(value) for value in output_tuple)
    return PythonFactorValidation(canonical_key=key, outputs=output_tuple)


def validate_reference(spec, level):
    """Validate this domain reference before any response is scored."""
    from modebench.validation import InputError, positive_integer
    parse_python_factor_spec(spec)
