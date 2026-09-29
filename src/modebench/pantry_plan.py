"""Executable nutrition-and-inventory plans for PantryPlan ModeBench.

The model emits a compact allocation such as ``lentils=150;rice=100``.  The
trusted specification supplies a frozen pantry, exact per-100g attributes,
dietary exclusions, and feasibility bounds.  Correctness and semantic identity
come from the same validation:

* every used ingredient exists, is allowed, and respects its available,
  minimum-serving, and quantity-step limits;
* exact decimal arithmetic checks mass, nutrition, cost, and other registered
  bounds; and
* the canonical mode is the sorted set of ingredients actually used.

Quantities affect correctness but not identity.  This deliberately collapses
nearby quantity variants of the same formulation instead of treating gram-level
perturbations as new semantic modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping


PANTRY_PLAN_VERIFIER = "pantry_plan"
PANTRY_PLAN_VERSION = "pantry-v1"

_ID = re.compile(r"[a-z][a-z0-9_]{0,31}")
_ATTRIBUTE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_TAG = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_MAX_INGREDIENTS = 16
_MAX_CANDIDATE_CHARS = 512
_ALLOWED_ATTRIBUTES = {
    "energy_kcal",
    "protein_g",
    "fiber_g",
    "fat_g",
    "carbohydrate_g",
    "sodium_mg",
    "cost_cents",
}


class PantryPlanError(ValueError):
    """Raised for an invalid trusted specification or candidate plan."""


@dataclass(frozen=True)
class PantryIngredient:
    ingredient_id: str
    available_g: int
    step_g: int
    min_if_used_g: int
    attributes_per_100g: tuple[tuple[str, Decimal], ...]
    tags: frozenset[str]

    @property
    def attributes(self) -> dict[str, Decimal]:
        return dict(self.attributes_per_100g)


@dataclass(frozen=True)
class PantryTarget:
    attribute: str
    minimum: Decimal | None
    maximum: Decimal | None


@dataclass(frozen=True)
class PantryPlanSpec:
    ingredients: tuple[PantryIngredient, ...]
    targets: tuple[PantryTarget, ...]
    min_ingredients: int
    max_ingredients: int
    forbidden_tags: frozenset[str]
    certified_mode_count: int

    @property
    def ingredient_by_id(self) -> dict[str, PantryIngredient]:
        return {row.ingredient_id: row for row in self.ingredients}


@dataclass(frozen=True)
class PantryPlanValidation:
    """One accepted allocation and its validator-derived semantic mode."""

    canonical_key: str
    allocations_g: tuple[tuple[str, int], ...]
    totals: tuple[tuple[str, Decimal], ...]


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise PantryPlanError(f"{label} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise PantryPlanError(f"{label} must be a finite decimal") from error
    if not result.is_finite() or result < 0:
        raise PantryPlanError(f"{label} must be a nonnegative finite decimal")
    return result


def _bounded_int(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PantryPlanError(f"{label} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise PantryPlanError(f"{label} is outside [{minimum}, {maximum}]")
    return result


def _tags(value: Any, *, label: str) -> frozenset[str]:
    if not isinstance(value, list) or len(value) > 32:
        raise PantryPlanError(f"{label} must be a bounded tag list")
    result: set[str] = set()
    for raw in value:
        tag = str(raw)
        if not _TAG.fullmatch(tag):
            raise PantryPlanError(f"{label} contains invalid tag {tag!r}")
        if tag in result:
            raise PantryPlanError(f"{label} contains duplicate tag {tag!r}")
        result.add(tag)
    return frozenset(result)


def parse_pantry_plan_spec(spec: Mapping[str, Any]) -> PantryPlanSpec:
    """Validate a trusted PantryPlan specification."""

    if spec.get("verifier") != PANTRY_PLAN_VERIFIER:
        raise PantryPlanError("wrong PantryPlan verifier")
    if spec.get("pantry_version") != PANTRY_PLAN_VERSION:
        raise PantryPlanError("unsupported PantryPlan version")

    raw_ingredients = spec.get("ingredients")
    if (
        not isinstance(raw_ingredients, list)
        or not 2 <= len(raw_ingredients) <= _MAX_INGREDIENTS
    ):
        raise PantryPlanError("ingredients must contain between 2 and 16 rows")

    ingredients: list[PantryIngredient] = []
    observed_ids: set[str] = set()
    for index, raw in enumerate(raw_ingredients):
        if not isinstance(raw, Mapping):
            raise PantryPlanError(f"ingredient {index} must be an object")
        ingredient_id = str(raw.get("id", ""))
        if not _ID.fullmatch(ingredient_id):
            raise PantryPlanError(f"ingredient {index} has an invalid id")
        if ingredient_id in observed_ids:
            raise PantryPlanError(f"duplicate ingredient id {ingredient_id!r}")
        observed_ids.add(ingredient_id)

        available_g = _bounded_int(
            raw.get("available_g"),
            label=f"{ingredient_id}.available_g",
            minimum=1,
            maximum=10_000,
        )
        step_g = _bounded_int(
            raw.get("step_g"),
            label=f"{ingredient_id}.step_g",
            minimum=1,
            maximum=1_000,
        )
        min_if_used_g = _bounded_int(
            raw.get("min_if_used_g"),
            label=f"{ingredient_id}.min_if_used_g",
            minimum=step_g,
            maximum=available_g,
        )
        if available_g % step_g or min_if_used_g % step_g:
            raise PantryPlanError(
                f"{ingredient_id} availability and minimum must align to step_g"
            )

        raw_attributes = raw.get("attributes_per_100g")
        if not isinstance(raw_attributes, Mapping) or not raw_attributes:
            raise PantryPlanError(
                f"{ingredient_id}.attributes_per_100g must be nonempty"
            )
        attributes: list[tuple[str, Decimal]] = []
        for raw_name, raw_value in raw_attributes.items():
            name = str(raw_name)
            if (
                not _ATTRIBUTE.fullmatch(name)
                or name not in _ALLOWED_ATTRIBUTES
            ):
                raise PantryPlanError(
                    f"{ingredient_id} has unsupported attribute {name!r}"
                )
            attributes.append(
                (
                    name,
                    _decimal(
                        raw_value,
                        label=f"{ingredient_id}.attributes_per_100g.{name}",
                    ),
                )
            )
        ingredients.append(
            PantryIngredient(
                ingredient_id=ingredient_id,
                available_g=available_g,
                step_g=step_g,
                min_if_used_g=min_if_used_g,
                attributes_per_100g=tuple(sorted(attributes)),
                tags=_tags(raw.get("tags", []), label=f"{ingredient_id}.tags"),
            )
        )

    min_ingredients = _bounded_int(
        spec.get("min_ingredients"),
        label="min_ingredients",
        minimum=1,
        maximum=len(ingredients),
    )
    max_ingredients = _bounded_int(
        spec.get("max_ingredients"),
        label="max_ingredients",
        minimum=min_ingredients,
        maximum=len(ingredients),
    )
    certified_mode_count = _bounded_int(
        spec.get("certified_mode_count"),
        label="certified_mode_count",
        minimum=2,
        maximum=1_000_000,
    )
    forbidden_tags = _tags(spec.get("forbidden_tags", []), label="forbidden_tags")

    raw_targets = spec.get("targets")
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise PantryPlanError("targets must be a nonempty object")
    targets: list[PantryTarget] = []
    for raw_name, raw_bounds in raw_targets.items():
        name = str(raw_name)
        if name != "mass_g" and name not in _ALLOWED_ATTRIBUTES:
            raise PantryPlanError(f"unsupported target attribute {name!r}")
        if not isinstance(raw_bounds, Mapping):
            raise PantryPlanError(f"target {name!r} must be an object")
        unknown_bounds = set(raw_bounds) - {"min", "max"}
        if unknown_bounds:
            raise PantryPlanError(f"target {name!r} has unknown bounds")
        minimum = (
            _decimal(raw_bounds["min"], label=f"targets.{name}.min")
            if "min" in raw_bounds
            else None
        )
        maximum = (
            _decimal(raw_bounds["max"], label=f"targets.{name}.max")
            if "max" in raw_bounds
            else None
        )
        if minimum is None and maximum is None:
            raise PantryPlanError(f"target {name!r} has no bound")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise PantryPlanError(f"target {name!r} has min greater than max")
        if name != "mass_g":
            for ingredient in ingredients:
                if name not in ingredient.attributes:
                    raise PantryPlanError(
                        f"ingredient {ingredient.ingredient_id!r} lacks target "
                        f"attribute {name!r}"
                    )
        targets.append(PantryTarget(name, minimum, maximum))

    return PantryPlanSpec(
        ingredients=tuple(ingredients),
        targets=tuple(sorted(targets, key=lambda target: target.attribute)),
        min_ingredients=min_ingredients,
        max_ingredients=max_ingredients,
        forbidden_tags=forbidden_tags,
        certified_mode_count=certified_mode_count,
    )


def parse_pantry_plan_candidate(candidate: str) -> tuple[tuple[str, int], ...]:
    """Parse ``ingredient_id=grams`` entries without executing model text."""

    text = str(candidate).strip()
    if not text or len(text) > _MAX_CANDIDATE_CHARS:
        raise PantryPlanError("candidate is empty or too long")
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    pieces = [piece.strip() for piece in re.split(r"[;,]", text)]
    if not pieces or any(not piece for piece in pieces):
        raise PantryPlanError("candidate contains an empty allocation")

    allocations: dict[str, int] = {}
    for piece in pieces:
        match = re.fullmatch(r"([a-z][a-z0-9_]{0,31})\s*=\s*([0-9]+)", piece)
        if match is None:
            raise PantryPlanError(f"invalid allocation {piece!r}")
        ingredient_id, raw_grams = match.groups()
        if ingredient_id in allocations:
            raise PantryPlanError(f"duplicate allocation for {ingredient_id!r}")
        grams = int(raw_grams)
        if not 1 <= grams <= 10_000:
            raise PantryPlanError(f"allocation for {ingredient_id!r} is out of range")
        allocations[ingredient_id] = grams
    return tuple(sorted(allocations.items()))


def validate_pantry_plan(
    candidate: str,
    spec: Mapping[str, Any],
) -> PantryPlanValidation | None:
    """Return the exact verified formulation mode, or ``None`` on rejection."""

    try:
        parsed_spec = parse_pantry_plan_spec(spec)
        allocations = parse_pantry_plan_candidate(candidate)
        if not (
            parsed_spec.min_ingredients
            <= len(allocations)
            <= parsed_spec.max_ingredients
        ):
            raise PantryPlanError("candidate uses the wrong number of ingredients")

        by_id = parsed_spec.ingredient_by_id
        totals: dict[str, Decimal] = {"mass_g": Decimal(0)}
        for ingredient_id, grams in allocations:
            ingredient = by_id.get(ingredient_id)
            if ingredient is None:
                raise PantryPlanError(f"unknown ingredient {ingredient_id!r}")
            if ingredient.tags & parsed_spec.forbidden_tags:
                raise PantryPlanError(f"ingredient {ingredient_id!r} is forbidden")
            if (
                grams < ingredient.min_if_used_g
                or grams > ingredient.available_g
                or grams % ingredient.step_g
            ):
                raise PantryPlanError(
                    f"allocation for {ingredient_id!r} violates pantry bounds"
                )
            grams_decimal = Decimal(grams)
            totals["mass_g"] += grams_decimal
            for name, per_100g in ingredient.attributes_per_100g:
                totals[name] = (
                    totals.get(name, Decimal(0))
                    + per_100g * grams_decimal / Decimal(100)
                )

        for target in parsed_spec.targets:
            value = totals.get(target.attribute, Decimal(0))
            if target.minimum is not None and value < target.minimum:
                raise PantryPlanError(f"{target.attribute} is below its minimum")
            if target.maximum is not None and value > target.maximum:
                raise PantryPlanError(f"{target.attribute} exceeds its maximum")

        support = tuple(ingredient_id for ingredient_id, _grams in allocations)
        canonical_key = f"{PANTRY_PLAN_VERIFIER}:{PANTRY_PLAN_VERSION}:" + "+".join(
            support
        )
        return PantryPlanValidation(
            canonical_key=canonical_key,
            allocations_g=allocations,
            totals=tuple(sorted(totals.items())),
        )
    except (PantryPlanError, ArithmeticError, ValueError):
        return None
