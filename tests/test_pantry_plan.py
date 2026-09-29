from __future__ import annotations

from decimal import Decimal
import json

from modebench.grading import (
    boxed_reward_fn,
    extract_normalized_final_answer,
    validated_modebench_exploration_identity,
    validated_modebench_outcome_key,
)
from modebench.pantry_plan import (
    PANTRY_PLAN_VERIFIER,
    PANTRY_PLAN_VERSION,
    parse_pantry_plan_spec,
    validate_pantry_plan,
)


def _ingredient(
    ingredient_id: str,
    *,
    energy: str,
    protein: str,
    fiber: str,
    sodium: str,
    available_g: int = 300,
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": ingredient_id,
        "available_g": available_g,
        "step_g": 25,
        "min_if_used_g": 50,
        "attributes_per_100g": {
            "energy_kcal": energy,
            "protein_g": protein,
            "fiber_g": fiber,
            "sodium_mg": sodium,
        },
        "tags": tags or [],
    }


def _spec() -> dict:
    return {
        "verifier": PANTRY_PLAN_VERIFIER,
        "pantry_version": PANTRY_PLAN_VERSION,
        "ingredients": [
            _ingredient(
                "lentils",
                energy="116",
                protein="9",
                fiber="7.9",
                sodium="2",
            ),
            _ingredient(
                "chickpeas",
                energy="164",
                protein="8.9",
                fiber="7.6",
                sodium="7",
            ),
            _ingredient(
                "brown_rice",
                energy="123",
                protein="2.7",
                fiber="1.6",
                sodium="4",
            ),
            _ingredient(
                "tofu",
                energy="144",
                protein="17.3",
                fiber="2.3",
                sodium="14",
                tags=["soy"],
            ),
        ],
        "targets": {
            "mass_g": {"min": "250", "max": "450"},
            "energy_kcal": {"min": "300", "max": "650"},
            "protein_g": {"min": "20"},
            "fiber_g": {"min": "10"},
            "sodium_mg": {"max": "100"},
        },
        "min_ingredients": 2,
        "max_ingredients": 3,
        "forbidden_tags": [],
        "certified_mode_count": 3,
    }


def test_pantry_plan_exactly_validates_inventory_and_nutrition():
    validation = validate_pantry_plan(
        "lentils=200;brown_rice=100",
        _spec(),
    )

    assert validation is not None
    assert validation.allocations_g == (("brown_rice", 100), ("lentils", 200))
    totals = dict(validation.totals)
    assert totals["mass_g"] == Decimal("300")
    assert totals["protein_g"] == Decimal("20.7")
    assert totals["fiber_g"] == Decimal("17.4")


def test_pantry_plan_mode_is_ingredient_support_not_quantity_or_order():
    first = validate_pantry_plan("lentils=200;brown_rice=100", _spec())
    second = validate_pantry_plan("brown_rice=125, lentils=200", _spec())
    distinct = validate_pantry_plan("chickpeas=200;brown_rice=100", _spec())

    assert first is not None and second is not None and distinct is not None
    assert first.canonical_key == second.canonical_key
    assert first.allocations_g != second.allocations_g
    assert first.canonical_key != distinct.canonical_key


def test_pantry_plan_rejects_alias_inflation_and_constraint_failures():
    spec = _spec()

    assert validate_pantry_plan("lentils=200;lentils=100", spec) is None
    assert validate_pantry_plan("lentils=210;brown_rice=100", spec) is None
    assert validate_pantry_plan("lentils=400;brown_rice=100", spec) is None
    assert validate_pantry_plan("brown_rice=150;chickpeas=100", spec) is None
    assert validate_pantry_plan("lentils=200;unknown=100", spec) is None


def test_pantry_plan_enforces_dietary_tags():
    spec = _spec()
    spec["forbidden_tags"] = ["soy"]

    assert validate_pantry_plan("tofu=150;brown_rice=100", spec) is None
    assert validate_pantry_plan("lentils=200;brown_rice=100", spec) is not None


def test_pantry_plan_requires_audited_multimode_spec():
    spec = _spec()
    spec["certified_mode_count"] = 1

    try:
        parse_pantry_plan_spec(spec)
    except ValueError as error:
        assert "certified_mode_count" in str(error)
    else:
        raise AssertionError("single-mode PantryPlan specification was accepted")


def test_pantry_plan_is_bound_into_modebench_reward_and_identity():
    reference = json.dumps(_spec())
    response = r"\boxed{lentils=200;brown_rice=100}"

    info, reward = boxed_reward_fn(response, reference)
    key = validated_modebench_outcome_key(response, reference)
    extracted = extract_normalized_final_answer(
        response,
        template="qwen_boxed",
        gt_answer=reference,
    )
    identity = validated_modebench_exploration_identity(response, reference)

    assert info == {"formatted": True}
    assert reward == 1.0
    assert key == "pantry_plan:pantry-v1:brown_rice+lentils"
    assert extracted == key
    assert identity is not None
    assert identity.endpoint_key == key
    assert identity.route_signature is None


def test_pantry_plan_grader_rejects_unverified_formulation():
    reference = json.dumps(_spec())
    response = r"\boxed{brown_rice=100;chickpeas=50}"

    info, reward = boxed_reward_fn(response, reference)

    assert info == {"formatted": True}
    assert reward == 0.0
    assert validated_modebench_outcome_key(response, reference) is None
