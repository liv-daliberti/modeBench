from __future__ import annotations

import json

import pytest

from modebench.grading import (
    boxed_reward_fn,
    validated_modebench_exploration_identity,
    validated_modebench_outcome_key,
)
from modebench.mathir import (
    MathIRError,
    enumerate_mathir_action_menu_keys,
    enumerate_mathir_action_menu_route_signatures,
    parse_mathir_action_program,
    validate_mathir_action_menu,
)


def _menu_spec(**updates):
    spec = {
        "verifier": "mathir_action_menu",
        "mathir_version": "linear-menu-v1",
        "bindings": {"a": 5, "b": 2, "c": 8, "d": 2},
        "initial_lhs": "add(mul(a,x),b)",
        "initial_rhs": "add(mul(d,x),c)",
        "max_steps": 4,
        "actions": {
            "A": "sub(b)",
            "B": "sub(mul(d,x))",
            "C": "div(sub(a,d))",
            "D": "sub(add(mul(d,x),b))",
            "E": "add(b)",
            "F": "div(a)",
        },
        "support_is_open": False,
        "num_completions": 5,
    }
    spec.update(updates)
    return spec


def test_action_menu_executes_exact_selected_combo_and_keys_states():
    spec = _menu_spec()

    constant_first = validate_mathir_action_menu("A;B;C", spec)
    variable_first = validate_mathir_action_menu("B;A;C", spec)
    combined = validate_mathir_action_menu("D;C", spec)

    assert constant_first is not None
    assert variable_first is not None
    assert combined is not None
    assert constant_first.solution == variable_first.solution == combined.solution == 2
    assert constant_first.action_ids == ("A", "B", "C")
    assert len(
        {
            constant_first.canonical_key,
            variable_first.canonical_key,
            combined.canonical_key,
        }
    ) == 3
    assert constant_first.canonical_key.startswith("mathir:linear-menu-v1:eq(")


def test_action_labels_do_not_define_identity():
    original = _menu_spec()
    relabeled = _menu_spec(
        actions={
            "A": "div(a)",
            "B": "add(b)",
            "C": "sub(add(mul(d,x),b))",
            "D": "div(sub(a,d))",
            "E": "sub(mul(d,x))",
            "F": "sub(b)",
        }
    )

    first = validate_mathir_action_menu("A;B;C", original)
    second = validate_mathir_action_menu("F;E;D", relabeled)

    assert first is not None
    assert second is not None
    assert first.canonical_key == second.canonical_key
    assert first.route_signature == second.route_signature


def test_route_signature_is_cross_prompt_and_coefficient_name_invariant():
    original = _menu_spec()
    relabeled = _menu_spec(
        bindings={"p": -7, "q": 11, "r": -3, "s": 2},
        initial_lhs="add(mul(p,x),q)",
        initial_rhs="add(mul(s,x),r)",
        actions={
            "A": "sub(q)",
            "B": "sub(mul(s,x))",
            "C": "div(sub(p,s))",
            "D": "sub(add(mul(s,x),q))",
            "E": "add(q)",
            "F": "div(p)",
        },
    )

    first = validate_mathir_action_menu("A;B;C", original)
    second = validate_mathir_action_menu("A;B;C", relabeled)

    assert first is not None
    assert second is not None
    assert first.canonical_key != second.canonical_key
    assert first.route_signature == second.route_signature
    assert first.route_signature.startswith("mathir-route:linear-route-v1:")
    for forbidden in ("A", "B", "C", "-7", "11"):
        assert forbidden not in first.route_signature


def test_distinct_executed_routes_have_distinct_route_signatures():
    spec = _menu_spec()
    constant_first = validate_mathir_action_menu("A;B;C", spec)
    variable_first = validate_mathir_action_menu("B;A;C", spec)
    combined = validate_mathir_action_menu("D;C", spec)

    assert constant_first is not None
    assert variable_first is not None
    assert combined is not None
    assert len(
        {
            constant_first.route_signature,
            variable_first.route_signature,
            combined.route_signature,
        }
    ) == 3


@pytest.mark.parametrize(
    "program",
    (
        "A;C",
        "A;B",
        "A;B;F",
        "A;B;C;E",
        "A;Z;C",
        "sub(b);B;C",
        "A;;B;C",
        "A;B;C;A;B",
        "I",
    ),
)
def test_action_menu_rejects_wrong_illegal_or_nonterminal_programs(program):
    assert validate_mathir_action_menu(program, _menu_spec()) is None


def test_action_program_surface_aliases_do_not_create_modes():
    spec = _menu_spec()
    plain = validate_mathir_action_menu("A;B;C", spec)
    spaced = validate_mathir_action_menu(" A ; B ; C ; ", spec)

    assert plain is not None
    assert spaced is not None
    assert plain.canonical_key == spaced.canonical_key


def test_action_menu_has_exact_finite_support():
    assert len(enumerate_mathir_action_menu_keys(_menu_spec())) == 5
    assert len(enumerate_mathir_action_menu_route_signatures(_menu_spec())) == 5


def test_action_menu_is_the_single_reward_and_bank_admission_boundary():
    reference = json.dumps(_menu_spec())
    info, reward = boxed_reward_fn(r"\boxed{A;B;C}", reference)
    key = validated_modebench_outcome_key(r"\boxed{A;B;C}", reference)

    assert info == {"formatted": True}
    assert reward == 1.0
    assert key is not None
    assert validated_modebench_outcome_key(r"\boxed{A;C}", reference) is None
    identity = validated_modebench_exploration_identity(
        r"\boxed{A;B;C}",
        reference,
    )
    assert identity is not None
    assert identity.endpoint_key == "mathir-solution:2/1"
    assert identity.route_signature is not None


def test_action_menu_reference_schema_fails_closed():
    for update in (
        {"mathir_version": "linear-menu-v2"},
        {"verifier": "mathir_algebra"},
        {"actions": {"A": "sub(b)", "C": "div(a)"}},
        {
            "actions": {
                "A": "sub(b)",
                "B": "sub(b)",
            }
        },
    ):
        assert validate_mathir_action_menu(
            "A;B;C",
            _menu_spec(**update),
        ) is None


def test_action_parser_accepts_only_bounded_menu_ids():
    assert parse_mathir_action_program(
        "A;C;",
        action_ids=("A", "B", "C"),
        max_steps=3,
    ) == ("A", "C")
    with pytest.raises(MathIRError):
        parse_mathir_action_program(
            "A;D",
            action_ids=("A", "B", "C"),
            max_steps=3,
        )
