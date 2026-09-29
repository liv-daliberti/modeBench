from __future__ import annotations

import json

import pytest

from modebench.grading import (
    boxed_reward_fn,
    extract_normalized_final_answer,
    validated_modebench_outcome_key,
)
from modebench.mathir import (
    MathIRError,
    parse_mathir_program,
    validate_mathir_algebra,
)


def _linear_spec(**updates):
    spec = {
        "verifier": "mathir_algebra",
        "mathir_version": "linear-v0",
        "bindings": {"a": 3, "b": 5, "c": 14},
        "initial_lhs": "add(mul(a,x),b)",
        "initial_rhs": "c",
        "max_steps": 4,
        "support_is_open": True,
    }
    spec.update(updates)
    return spec


def test_mathir_reward_executes_program_and_keys_the_executed_states():
    reference = json.dumps(_linear_spec())

    info, reward = boxed_reward_fn(
        r"\boxed{sub(b);div(a)}",
        reference,
    )
    key = validated_modebench_outcome_key(
        r"\boxed{sub(b);div(a)}",
        reference,
    )

    assert info == {"formatted": True}
    assert reward == 1.0
    assert key is not None
    assert key.startswith("mathir:linear-v0:eq(")
    assert key.count(">") == 1
    assert extract_normalized_final_answer(
        r"\boxed{sub(b);div(a)}",
        template="qwen_boxed",
        gt_answer=reference,
    ) == key


def test_mathir_distinguishes_valid_state_paths_not_surface_aliases():
    spec = _linear_spec()

    direct = validate_mathir_algebra("sub(b);div(a)", spec)
    alias = validate_mathir_algebra("add(neg(b));div(a)", spec)
    divide_first = validate_mathir_algebra(
        "div(a);sub(div(b,a))",
        spec,
    )

    assert direct is not None
    assert alias is not None
    assert divide_first is not None
    assert direct.canonical_key == alias.canonical_key
    assert divide_first.canonical_key != direct.canonical_key
    assert direct.solution == divide_first.solution


@pytest.mark.parametrize(
    "program",
    (
        "x=3",
        "sub(5);div(a)",
        "__import__(a)",
        "sub(z);div(a)",
        "sub(b);div(c)",
        "mul(x);sub(b);div(a)",
        "sub(div(a,x));div(a)",
        "sub(b);div(a);add(b);sub(b)",
    ),
)
def test_mathir_rejects_nonlanguage_wrong_or_nonreversible_programs(program):
    assert validate_mathir_algebra(program, _linear_spec()) is None


def test_mathir_rejects_cycles_even_when_every_command_is_algebraic():
    spec = _linear_spec(max_steps=4)

    # add(a) and sub(a) exact-normalize back to the initial equation.
    assert validate_mathir_algebra(
        "add(a);sub(a);sub(b);div(a)",
        spec,
    ) is None


def test_mathir_canonicalizes_terminal_semicolon_and_program_only_fence():
    reference = json.dumps(_linear_spec())
    plain = validated_modebench_outcome_key("sub(b);div(a)", reference)

    assert validated_modebench_outcome_key("sub(b);div(a);", reference) == plain
    assert validated_modebench_outcome_key(
        "```mathir\nsub(b); div(a)\n```",
        reference,
    ) == plain


def test_mathir_rejects_zero_division_from_the_concrete_formal_environment():
    spec = _linear_spec(bindings={"a": 3, "b": 5, "c": 14, "d": 3})
    spec["initial_rhs"] = "add(c,d)"

    assert validate_mathir_algebra("div(sub(a,d))", spec) is None


def test_mathir_two_sided_linear_equation_has_multiple_real_paths():
    spec = {
        "verifier": "mathir_algebra",
        "mathir_version": "linear-v0",
        "bindings": {"a": 5, "b": 2, "c": 8, "d": 2},
        "initial_lhs": "add(mul(a,x),b)",
        "initial_rhs": "add(mul(d,x),c)",
        "max_steps": 4,
        "support_is_open": True,
    }

    move_x_first = validate_mathir_algebra(
        "sub(mul(d,x));sub(b);div(sub(a,d))",
        spec,
    )
    move_constant_first = validate_mathir_algebra(
        "sub(b);sub(mul(d,x));div(sub(a,d))",
        spec,
    )
    combined = validate_mathir_algebra(
        "sub(add(mul(d,x),b));div(sub(a,d))",
        spec,
    )

    assert move_x_first is not None
    assert move_constant_first is not None
    assert combined is not None
    assert {
        move_x_first.canonical_key,
        move_constant_first.canonical_key,
        combined.canonical_key,
    }.__len__() == 3
    assert {move_x_first.solution, move_constant_first.solution, combined.solution} == {
        2
    }


def test_mathir_reference_schema_is_fail_closed():
    for update in (
        {"mathir_version": "linear-v1"},
        {"bindings": {"a": 3, "b": 5}},
        {"initial_lhs": "sqrt(x)"},
        {"max_steps": 100},
    ):
        assert validate_mathir_algebra(
            "sub(b);div(a)",
            _linear_spec(**update),
        ) is None


def test_model_parser_has_no_numeric_literal_or_python_escape_hatch():
    with pytest.raises(MathIRError):
        parse_mathir_program(
            "sub(5);div(a)",
            allowed_symbols={"a", "b", "c", "x"},
            max_steps=4,
        )
    with pytest.raises(MathIRError):
        parse_mathir_program(
            "eval(a)",
            allowed_symbols={"a", "b", "c", "x"},
            max_steps=4,
        )
