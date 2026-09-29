"""A small, fail-closed executable algebra language for canonical search.

MathIR linear v0 is intentionally narrower than ordinary mathematical text.
The model emits a semicolon-separated sequence of equation transformations,
for example ``sub(b);div(a)``.  Every command is applied to both sides of the
current equation and exact rational normalization happens after every step.

The validator and canonicalizer share one execution path: a canonical strategy
key is produced only from the normalized states created by a successful
execution.  There is no parser for prose, LaTeX derivations, Python, or a
model-supplied final answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
import re
from typing import Any, Iterable, Mapping

import sympy


MATHIR_VERIFIER = "mathir_algebra"
MATHIR_VERSION = "linear-v0"
MATHIR_MENU_VERIFIER = "mathir_action_menu"
MATHIR_MENU_VERSION = "linear-menu-v1"
MATHIR_ROUTE_VERSION = "linear-route-v1"
_MAX_REFERENCE_SYMBOLS = 6
_MAX_PROGRAM_STEPS = 4
_MAX_PROGRAM_CHARS = 160
_MAX_ARGUMENT_NODES = 11
_MAX_ARGUMENT_DEPTH = 6
_MAX_MENU_ACTIONS = 8
_MODEL_OPERATORS = frozenset({"add", "sub", "mul", "div", "neg"})
_COMMANDS = frozenset({"add", "sub", "mul", "div"})
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[(),;]")
_MENU_ACTION_RE = re.compile(r"[A-H]")


@dataclass(frozen=True)
class Expr:
    """A bounded MathIR expression.

    ``const`` nodes are interpreter-internal exact rationals.  The model-side
    parser never accepts numeric literals.
    """

    op: str
    args: tuple["Expr", ...] = ()
    value: str | Fraction | None = None


@dataclass(frozen=True)
class Command:
    op: str
    argument: Expr


@dataclass(frozen=True)
class EquationState:
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class MathIRValidation:
    canonical_key: str
    solution: Fraction
    commands: tuple[Command, ...]
    states: tuple[EquationState, ...]
    action_ids: tuple[str, ...] = ()
    route_signature: str = ""


class MathIRError(ValueError):
    """Raised for a malformed or invalid MathIR program."""


class _ExpressionParser:
    def __init__(
        self,
        tokens: list[str],
        *,
        allowed_symbols: frozenset[str],
        allow_constants: bool,
    ) -> None:
        self.tokens = tokens
        self.index = 0
        self.allowed_symbols = allowed_symbols
        self.allow_constants = bool(allow_constants)

    def _take(self, expected: str | None = None) -> str:
        if self.index >= len(self.tokens):
            raise MathIRError("unexpected end of expression")
        token = self.tokens[self.index]
        if expected is not None and token != expected:
            raise MathIRError(f"expected {expected!r}")
        self.index += 1
        return token

    def parse(self, *, depth: int = 0) -> Expr:
        if depth > _MAX_ARGUMENT_DEPTH:
            raise MathIRError("expression nesting is too deep")
        token = self._take()
        if token in {"(", ")", ",", ";"}:
            raise MathIRError("expected a symbol or operator")
        if self.index < len(self.tokens) and self.tokens[self.index] == "(":
            if token not in _MODEL_OPERATORS:
                raise MathIRError(f"unsupported operator {token!r}")
            self._take("(")
            first = self.parse(depth=depth + 1)
            if token == "neg":
                self._take(")")
                return Expr("neg", (first,))
            self._take(",")
            second = self.parse(depth=depth + 1)
            self._take(")")
            return Expr(token, (first, second))
        if token in self.allowed_symbols:
            return Expr("symbol", value=token)
        if self.allow_constants and re.fullmatch(r"-?\d+(?:/\d+)?", token):
            return Expr("const", value=Fraction(token))
        raise MathIRError(f"unknown symbol {token!r}")


def _tokenize(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", str(text))
    if not compact:
        raise MathIRError("empty MathIR text")
    tokens = _TOKEN_RE.findall(compact)
    if "".join(tokens) != compact:
        raise MathIRError("unsupported MathIR character or numeric literal")
    return tokens


def parse_mathir_expression(
    text: str,
    *,
    allowed_symbols: Iterable[str],
) -> Expr:
    """Parse one model-authored expression without using Python evaluation."""

    tokens = _tokenize(text)
    parser = _ExpressionParser(
        tokens,
        allowed_symbols=frozenset(str(symbol) for symbol in allowed_symbols),
        allow_constants=False,
    )
    expression = parser.parse()
    if parser.index != len(tokens):
        raise MathIRError("trailing expression tokens")
    if _expr_node_count(expression) > _MAX_ARGUMENT_NODES:
        raise MathIRError("expression is too large")
    return expression


def _parse_trusted_expression(
    text: str,
    *,
    allowed_symbols: Iterable[str],
) -> Expr:
    """Parse a dataset-owned formal expression.

    Dataset expressions currently use no constants, but this separate entry
    point makes the trust boundary explicit and permits exact rationals if a
    later, versioned reference schema needs them.
    """

    tokens = _tokenize(text)
    parser = _ExpressionParser(
        tokens,
        allowed_symbols=frozenset(str(symbol) for symbol in allowed_symbols),
        allow_constants=True,
    )
    expression = parser.parse()
    if parser.index != len(tokens):
        raise MathIRError("trailing trusted-expression tokens")
    if _expr_node_count(expression) > 31:
        raise MathIRError("trusted expression is too large")
    return expression


def parse_mathir_program(
    text: str,
    *,
    allowed_symbols: Iterable[str],
    max_steps: int,
) -> tuple[Command, ...]:
    """Parse a bounded sequence such as ``sub(b);div(a)``."""

    compact = re.sub(r"\s+", "", str(text))
    if not compact or len(compact) > _MAX_PROGRAM_CHARS:
        raise MathIRError("program is empty or too long")
    # A final statement terminator is surface formatting, not a new action.
    compact = compact[:-1] if compact.endswith(";") else compact
    if not compact or compact.startswith(";") or ";;" in compact:
        raise MathIRError("empty program command")
    command_texts = compact.split(";")
    if not 1 <= len(command_texts) <= int(max_steps):
        raise MathIRError("program has an invalid number of commands")
    commands: list[Command] = []
    for command_text in command_texts:
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\((.*)\)", command_text)
        if match is None:
            raise MathIRError("commands must use op(expression) syntax")
        op, argument_text = match.groups()
        if op not in _COMMANDS:
            raise MathIRError(f"unsupported command {op!r}")
        argument = parse_mathir_expression(
            argument_text,
            allowed_symbols=allowed_symbols,
        )
        commands.append(Command(op, argument))
    return tuple(commands)


def _expr_node_count(expression: Expr) -> int:
    return 1 + sum(_expr_node_count(argument) for argument in expression.args)


def _expr_symbols(expression: Expr) -> set[str]:
    if expression.op == "symbol":
        assert isinstance(expression.value, str)
        return {expression.value}
    return set().union(*(_expr_symbols(argument) for argument in expression.args), set())


def _fraction_from_reference(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise MathIRError("boolean binding")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str) and re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", value.strip()):
        return Fraction(value.strip())
    raise MathIRError("bindings must be exact integers or rational strings")


def _expr_to_sympy(expression: Expr) -> sympy.Expr:
    if expression.op == "symbol":
        assert isinstance(expression.value, str)
        return sympy.Symbol(expression.value)
    if expression.op == "const":
        assert isinstance(expression.value, Fraction)
        return sympy.Rational(expression.value.numerator, expression.value.denominator)
    converted = tuple(_expr_to_sympy(argument) for argument in expression.args)
    if expression.op == "add":
        return converted[0] + converted[1]
    if expression.op == "sub":
        return converted[0] - converted[1]
    if expression.op == "mul":
        return converted[0] * converted[1]
    if expression.op == "div":
        return converted[0] / converted[1]
    if expression.op == "neg":
        return -converted[0]
    if expression.op == "inv":
        return sympy.Integer(1) / converted[0]
    raise MathIRError(f"unsupported internal expression {expression.op!r}")


def _fold(op: str, arguments: tuple[Expr, ...]) -> Expr:
    if not arguments:
        return Expr("const", value=Fraction(0 if op == "add" else 1, 1))
    result = arguments[0]
    for argument in arguments[1:]:
        result = Expr(op, (result, argument))
    return result


def _expr_from_sympy(expression: sympy.Expr) -> Expr:
    if expression.is_Symbol:
        return Expr("symbol", value=str(expression))
    if expression.is_Rational:
        return Expr(
            "const",
            value=Fraction(int(expression.p), int(expression.q)),
        )
    if expression.is_Add:
        return _fold(
            "add",
            tuple(_expr_from_sympy(argument) for argument in expression.args),
        )
    if expression.is_Mul:
        return _fold(
            "mul",
            tuple(_expr_from_sympy(argument) for argument in expression.args),
        )
    if expression.is_Pow and expression.exp == -1:
        return Expr("inv", (_expr_from_sympy(expression.base),))
    raise MathIRError(f"normalizer produced unsupported expression {expression!r}")


def _normalize_expr(expression: Expr) -> Expr:
    symbolic = _expr_to_sympy(expression)
    normalized = sympy.cancel(symbolic)
    return _expr_from_sympy(normalized)


def _canonical_parts(expression: Expr) -> tuple[str, ...]:
    if expression.op not in {"add", "mul"}:
        return (_canonical_expr(expression),)
    parts: list[str] = []
    for argument in expression.args:
        converted = _canonicalized_expr(argument)
        if converted.op == expression.op:
            parts.extend(_canonical_parts(converted))
        else:
            parts.append(_canonical_expr(converted))
    return tuple(sorted(parts))


def _canonicalized_expr(expression: Expr) -> Expr:
    if expression.op == "sub":
        return Expr(
            "add",
            (
                _canonicalized_expr(expression.args[0]),
                Expr("neg", (_canonicalized_expr(expression.args[1]),)),
            ),
        )
    if expression.op == "div":
        return Expr(
            "mul",
            (
                _canonicalized_expr(expression.args[0]),
                Expr("inv", (_canonicalized_expr(expression.args[1]),)),
            ),
        )
    return Expr(
        expression.op,
        tuple(_canonicalized_expr(argument) for argument in expression.args),
        expression.value,
    )


def _canonical_expr(expression: Expr) -> str:
    expression = _canonicalized_expr(expression)
    if expression.op == "symbol":
        assert isinstance(expression.value, str)
        return expression.value
    if expression.op == "const":
        assert isinstance(expression.value, Fraction)
        if expression.value.denominator == 1:
            return str(expression.value.numerator)
        return f"rat({expression.value.numerator},{expression.value.denominator})"
    if expression.op in {"add", "mul"}:
        return f"{expression.op}({','.join(_canonical_parts(expression))})"
    if expression.op in {"neg", "inv"}:
        return f"{expression.op}({_canonical_expr(expression.args[0])})"
    raise MathIRError(f"cannot canonicalize {expression.op!r}")


def _canonical_state(state: EquationState) -> str:
    return f"eq({_canonical_expr(state.lhs)},{_canonical_expr(state.rhs)})"


def _rename_expr_symbols(
    expression: Expr,
    symbol_map: Mapping[str, str],
) -> Expr:
    if expression.op == "symbol":
        assert isinstance(expression.value, str)
        return Expr(
            "symbol",
            value=symbol_map.get(expression.value, expression.value),
        )
    return Expr(
        expression.op,
        tuple(
            _rename_expr_symbols(argument, symbol_map)
            for argument in expression.args
        ),
        expression.value,
    )


def _alpha_canonical_route(
    initial_state: EquationState,
    commands: tuple[Command, ...],
) -> str:
    """Canonicalize a verified route independently of coefficient names.

    At most six coefficient symbols are allowed by the reference schema, so a
    small exhaustive alpha-renaming is simpler and safer than relying on
    symbol-name or traversal-order heuristics. Numeric binding values never
    enter this representation.
    """

    symbols = sorted(
        (
            _expr_symbols(initial_state.lhs)
            | _expr_symbols(initial_state.rhs)
            | set().union(
                *(_expr_symbols(command.argument) for command in commands),
                set(),
            )
        )
        - {"x"}
    )
    roles = tuple(f"c{index}" for index in range(len(symbols)))
    candidates: list[str] = []
    for assigned_symbols in permutations(symbols):
        symbol_map = {
            symbol: role for symbol, role in zip(assigned_symbols, roles)
        }
        renamed_initial = EquationState(
            _rename_expr_symbols(initial_state.lhs, symbol_map),
            _rename_expr_symbols(initial_state.rhs, symbol_map),
        )
        command_parts = []
        for command in commands:
            renamed_argument = _rename_expr_symbols(
                command.argument,
                symbol_map,
            )
            command_parts.append(
                f"{command.op}({_canonical_expr(renamed_argument)})"
            )
        candidates.append(
            f"init={_canonical_state(renamed_initial)}"
            f"|commands={'>'.join(command_parts)}"
        )
    if not candidates:
        candidates.append(
            f"init={_canonical_state(initial_state)}"
            f"|commands={'>'.join(command.op for command in commands)}"
        )
    return f"mathir-route:{MATHIR_ROUTE_VERSION}:{min(candidates)}"


def _validate_denominators(
    expression: Expr,
    *,
    bindings: Mapping[str, Fraction],
) -> None:
    if expression.op == "div":
        denominator = expression.args[1]
        if "x" in _expr_symbols(denominator):
            raise MathIRError("x-dependent denominators are not supported")
        if _eval_fraction(denominator, bindings) == 0:
            raise MathIRError("division by zero in command expression")
    for argument in expression.args:
        _validate_denominators(argument, bindings=bindings)


def _eval_fraction(
    expression: Expr,
    bindings: Mapping[str, Fraction],
) -> Fraction:
    if expression.op == "symbol":
        assert isinstance(expression.value, str)
        if expression.value not in bindings:
            raise MathIRError("cannot evaluate an expression containing x")
        return bindings[expression.value]
    if expression.op == "const":
        assert isinstance(expression.value, Fraction)
        return expression.value
    values = tuple(_eval_fraction(argument, bindings) for argument in expression.args)
    if expression.op == "add":
        return values[0] + values[1]
    if expression.op == "sub":
        return values[0] - values[1]
    if expression.op == "mul":
        return values[0] * values[1]
    if expression.op == "div":
        if values[1] == 0:
            raise MathIRError("division by zero")
        return values[0] / values[1]
    if expression.op == "neg":
        return -values[0]
    if expression.op == "inv":
        if values[0] == 0:
            raise MathIRError("division by zero")
        return Fraction(1, 1) / values[0]
    raise MathIRError(f"cannot evaluate {expression.op!r}")


def _initial_solution(
    state: EquationState,
    *,
    bindings: Mapping[str, Fraction],
) -> Fraction:
    x = sympy.Symbol("x")
    substitutions = {
        sympy.Symbol(name): sympy.Rational(value.numerator, value.denominator)
        for name, value in bindings.items()
    }
    equation = sympy.cancel(
        (_expr_to_sympy(state.lhs) - _expr_to_sympy(state.rhs)).subs(substitutions)
    )
    numerator, denominator = sympy.together(equation).as_numer_denom()
    if x in denominator.free_symbols:
        raise MathIRError("initial equation has an x-dependent denominator")
    polynomial = sympy.Poly(sympy.expand(numerator), x)
    if polynomial.degree() != 1:
        raise MathIRError("initial equation is not uniquely linear")
    coefficient = polynomial.coeff_monomial(x)
    constant = polynomial.coeff_monomial(1)
    if coefficient == 0:
        raise MathIRError("initial equation has no unique solution")
    solution = sympy.cancel(-constant / coefficient)
    if not solution.is_Rational:
        raise MathIRError("initial solution is not rational")
    return Fraction(int(solution.p), int(solution.q))


def _apply_command(
    state: EquationState,
    command: Command,
    *,
    bindings: Mapping[str, Fraction],
) -> EquationState:
    _validate_denominators(command.argument, bindings=bindings)
    argument_symbols = _expr_symbols(command.argument)
    if command.op in {"mul", "div"}:
        if "x" in argument_symbols:
            raise MathIRError("multiplication and division by x are not reversible")
        if _eval_fraction(command.argument, bindings) == 0:
            raise MathIRError("multiplication and division require a nonzero argument")
    if command.op == "add":
        lhs = Expr("add", (state.lhs, command.argument))
        rhs = Expr("add", (state.rhs, command.argument))
    elif command.op == "sub":
        lhs = Expr("sub", (state.lhs, command.argument))
        rhs = Expr("sub", (state.rhs, command.argument))
    elif command.op == "mul":
        lhs = Expr("mul", (state.lhs, command.argument))
        rhs = Expr("mul", (state.rhs, command.argument))
    elif command.op == "div":
        lhs = Expr("div", (state.lhs, command.argument))
        rhs = Expr("div", (state.rhs, command.argument))
    else:
        raise MathIRError(f"unsupported command {command.op!r}")
    # This exact normalizer is part of the interpreter semantics, rather than
    # model-authored text which could claim a simplification without doing it.
    return EquationState(_normalize_expr(lhs), _normalize_expr(rhs))


def _validated_reference(
    spec: Mapping[str, Any],
) -> tuple[EquationState, dict[str, Fraction], int]:
    if spec.get("verifier") != MATHIR_VERIFIER:
        raise MathIRError("wrong verifier")
    if spec.get("mathir_version") != MATHIR_VERSION:
        raise MathIRError("unsupported MathIR version")
    raw_bindings = spec.get("bindings")
    if not isinstance(raw_bindings, dict):
        raise MathIRError("missing bindings")
    if not 1 <= len(raw_bindings) <= _MAX_REFERENCE_SYMBOLS:
        raise MathIRError("invalid number of bindings")
    bindings: dict[str, Fraction] = {}
    for raw_name, raw_value in raw_bindings.items():
        name = str(raw_name)
        if not re.fullmatch(r"[a-wyz]", name) or name == "x":
            raise MathIRError("binding names must be single lowercase coefficient symbols")
        bindings[name] = _fraction_from_reference(raw_value)
    if len(bindings) != len(raw_bindings):
        raise MathIRError("duplicate binding names")
    max_steps = int(spec.get("max_steps", _MAX_PROGRAM_STEPS))
    if not 1 <= max_steps <= _MAX_PROGRAM_STEPS:
        raise MathIRError("invalid max_steps")
    allowed_symbols = frozenset(bindings) | {"x"}
    lhs = _parse_trusted_expression(
        str(spec["initial_lhs"]),
        allowed_symbols=allowed_symbols,
    )
    rhs = _parse_trusted_expression(
        str(spec["initial_rhs"]),
        allowed_symbols=allowed_symbols,
    )
    referenced_coefficients = (_expr_symbols(lhs) | _expr_symbols(rhs)) - {"x"}
    if referenced_coefficients != set(bindings):
        raise MathIRError("bindings and initial equation symbols disagree")
    state = EquationState(_normalize_expr(lhs), _normalize_expr(rhs))
    _initial_solution(state, bindings=bindings)
    return state, bindings, max_steps


def _execute_mathir_commands(
    *,
    initial_state: EquationState,
    bindings: Mapping[str, Fraction],
    commands: tuple[Command, ...],
    key_version: str,
    action_ids: tuple[str, ...] = (),
) -> MathIRValidation:
    target_solution = _initial_solution(initial_state, bindings=bindings)
    seen = {_canonical_state(initial_state)}
    states: list[EquationState] = []
    state = initial_state
    for command in commands:
        state = _apply_command(state, command, bindings=bindings)
        state_key = _canonical_state(state)
        if state_key in seen:
            raise MathIRError("program revisits a previous equation state")
        seen.add(state_key)
        states.append(state)

    if state.lhs == Expr("symbol", value="x"):
        final_expression = state.rhs
    elif state.rhs == Expr("symbol", value="x"):
        final_expression = state.lhs
    else:
        raise MathIRError("program does not finish with x isolated")
    if "x" in _expr_symbols(final_expression):
        raise MathIRError("final expression still contains x")
    solution = _eval_fraction(final_expression, bindings)
    if solution != target_solution:
        raise MathIRError("executed program has the wrong solution")
    canonical_key = (
        f"mathir:{key_version}:"
        + ">".join(_canonical_state(executed_state) for executed_state in states)
    )
    route_signature = _alpha_canonical_route(initial_state, commands)
    return MathIRValidation(
        canonical_key=canonical_key,
        solution=solution,
        commands=commands,
        states=tuple(states),
        action_ids=action_ids,
        route_signature=route_signature,
    )


def validate_mathir_algebra(
    program_text: str,
    spec: Mapping[str, Any],
) -> MathIRValidation | None:
    """Execute and validate a MathIR program, returning its canonical path.

    All failures return ``None``.  This function is the single admission
    boundary used by both task reward and the online canonical bank.
    """

    try:
        initial_state, bindings, max_steps = _validated_reference(spec)
        allowed_symbols = frozenset(bindings) | {"x"}
        commands = parse_mathir_program(
            program_text,
            allowed_symbols=allowed_symbols,
            max_steps=max_steps,
        )
        return _execute_mathir_commands(
            initial_state=initial_state,
            bindings=bindings,
            commands=commands,
            key_version=MATHIR_VERSION,
        )
    except Exception:
        return None


def _validated_menu_reference(
    spec: Mapping[str, Any],
) -> tuple[
    EquationState,
    dict[str, Fraction],
    int,
    dict[str, Command],
]:
    if spec.get("verifier") != MATHIR_MENU_VERIFIER:
        raise MathIRError("wrong menu verifier")
    if spec.get("mathir_version") != MATHIR_MENU_VERSION:
        raise MathIRError("unsupported menu MathIR version")
    base_spec = dict(spec)
    base_spec["verifier"] = MATHIR_VERIFIER
    base_spec["mathir_version"] = MATHIR_VERSION
    initial_state, bindings, max_steps = _validated_reference(base_spec)
    raw_actions = spec.get("actions")
    if not isinstance(raw_actions, dict):
        raise MathIRError("missing action menu")
    if not 2 <= len(raw_actions) <= _MAX_MENU_ACTIONS:
        raise MathIRError("invalid action menu size")
    expected_ids = [chr(ord("A") + index) for index in range(len(raw_actions))]
    if list(raw_actions) != expected_ids:
        raise MathIRError("action IDs must be contiguous and ordered")
    allowed_symbols = frozenset(bindings) | {"x"}
    actions: dict[str, Command] = {}
    normalized_programs: set[str] = set()
    for action_id, raw_program in raw_actions.items():
        if _MENU_ACTION_RE.fullmatch(str(action_id)) is None:
            raise MathIRError("invalid action ID")
        program = re.sub(r"\s+", "", str(raw_program))
        if program in normalized_programs:
            raise MathIRError("duplicate action semantics")
        parsed = parse_mathir_program(
            program,
            allowed_symbols=allowed_symbols,
            max_steps=1,
        )
        if len(parsed) != 1:
            raise MathIRError("each action must contain exactly one command")
        normalized_programs.add(program)
        actions[str(action_id)] = parsed[0]
    return initial_state, bindings, max_steps, actions


def parse_mathir_action_program(
    text: str,
    *,
    action_ids: Iterable[str],
    max_steps: int,
) -> tuple[str, ...]:
    """Parse a bounded sequence of prompt-local action IDs."""

    compact = re.sub(r"\s+", "", str(text))
    if not compact or len(compact) > _MAX_PROGRAM_CHARS:
        raise MathIRError("action program is empty or too long")
    compact = compact[:-1] if compact.endswith(";") else compact
    if not compact or compact.startswith(";") or ";;" in compact:
        raise MathIRError("empty action")
    selected = tuple(compact.split(";"))
    if not 1 <= len(selected) <= int(max_steps):
        raise MathIRError("action program has an invalid number of steps")
    allowed = frozenset(str(action_id) for action_id in action_ids)
    if any(
        _MENU_ACTION_RE.fullmatch(action_id) is None or action_id not in allowed
        for action_id in selected
    ):
        raise MathIRError("unknown action ID")
    return selected


def validate_mathir_action_menu(
    program_text: str,
    spec: Mapping[str, Any],
) -> MathIRValidation | None:
    """Execute the exact prompt-local action sequence and key its state path."""

    try:
        initial_state, bindings, max_steps, actions = _validated_menu_reference(spec)
        action_ids = parse_mathir_action_program(
            program_text,
            action_ids=actions,
            max_steps=max_steps,
        )
        commands = tuple(actions[action_id] for action_id in action_ids)
        return _execute_mathir_commands(
            initial_state=initial_state,
            bindings=bindings,
            commands=commands,
            key_version=MATHIR_MENU_VERSION,
            action_ids=action_ids,
        )
    except Exception:
        return None


def enumerate_mathir_action_menu_keys(
    spec: Mapping[str, Any],
) -> set[str]:
    """Exhaustively enumerate the bounded menu's distinct verified state paths."""

    return {
        validation.canonical_key
        for validation in enumerate_mathir_action_menu_validations(spec)
    }


def _terminal_solution(
    state: EquationState,
    *,
    bindings: Mapping[str, Fraction],
    target_solution: Fraction,
) -> Fraction | None:
    if state.lhs == Expr("symbol", value="x"):
        final_expression = state.rhs
    elif state.rhs == Expr("symbol", value="x"):
        final_expression = state.lhs
    else:
        return None
    if "x" in _expr_symbols(final_expression):
        return None
    solution = _eval_fraction(final_expression, bindings)
    return solution if solution == target_solution else None


def enumerate_mathir_action_menu_validations(
    spec: Mapping[str, Any],
) -> tuple[MathIRValidation, ...]:
    """Enumerate exact support while caching deterministic state transitions."""

    initial_state, bindings, max_steps, actions = _validated_menu_reference(spec)
    target_solution = _initial_solution(initial_state, bindings=bindings)
    transition_cache: dict[
        tuple[str, str], tuple[EquationState, str] | None
    ] = {}
    admitted: dict[str, MathIRValidation] = {}

    def transition(
        state: EquationState,
        action_id: str,
    ) -> tuple[EquationState, str] | None:
        state_key = _canonical_state(state)
        cache_key = (state_key, action_id)
        if cache_key not in transition_cache:
            try:
                next_state = _apply_command(
                    state,
                    actions[action_id],
                    bindings=bindings,
                )
                transition_cache[cache_key] = (
                    next_state,
                    _canonical_state(next_state),
                )
            except Exception:
                transition_cache[cache_key] = None
        return transition_cache[cache_key]

    def visit(
        state: EquationState,
        *,
        seen: frozenset[str],
        commands: tuple[Command, ...],
        action_ids: tuple[str, ...],
        states: tuple[EquationState, ...],
    ) -> None:
        if len(commands) >= max_steps:
            return
        for action_id in actions:
            result = transition(state, action_id)
            if result is None:
                continue
            next_state, next_state_key = result
            if next_state_key in seen:
                continue
            next_commands = commands + (actions[action_id],)
            next_action_ids = action_ids + (action_id,)
            next_states = states + (next_state,)
            solution = _terminal_solution(
                next_state,
                bindings=bindings,
                target_solution=target_solution,
            )
            if solution is not None:
                canonical_key = (
                    f"mathir:{MATHIR_MENU_VERSION}:"
                    + ">".join(
                        _canonical_state(executed_state)
                        for executed_state in next_states
                    )
                )
                admitted[canonical_key] = MathIRValidation(
                    canonical_key=canonical_key,
                    solution=solution,
                    commands=next_commands,
                    states=next_states,
                    action_ids=next_action_ids,
                    route_signature=_alpha_canonical_route(
                        initial_state,
                        next_commands,
                    ),
                )
            visit(
                next_state,
                seen=seen | {next_state_key},
                commands=next_commands,
                action_ids=next_action_ids,
                states=next_states,
            )

    initial_key = _canonical_state(initial_state)
    visit(
        initial_state,
        seen=frozenset({initial_key}),
        commands=(),
        action_ids=(),
        states=(),
    )
    return tuple(admitted[key] for key in sorted(admitted))


def enumerate_mathir_action_menu_route_signatures(
    spec: Mapping[str, Any],
) -> set[str]:
    """Exhaustively enumerate the menu's verified cross-prompt route support."""

    return {
        validation.route_signature
        for validation in enumerate_mathir_action_menu_validations(spec)
    }


def certified_mathir_strategy_keys(
    spec: Mapping[str, Any],
    programs: Iterable[str],
) -> set[str]:
    """Validate a finite audit list without treating it as exhaustive support."""

    keys: set[str] = set()
    for program in programs:
        validation = validate_mathir_algebra(program, spec)
        if validation is None:
            raise MathIRError(f"certified program failed validation: {program}")
        keys.add(validation.canonical_key)
    return keys


def mathir_command_histogram(validation: MathIRValidation) -> Counter[str]:
    """Small diagnostic helper used by audits and tests."""

    return Counter(command.op for command in validation.commands)
