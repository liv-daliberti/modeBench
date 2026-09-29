"""Development adapter for support-first PantryPlan actions.

The language policy chooses only the ingredient support.  A deterministic,
trusted environment transition searches the registered quantity grid for the
lexicographically first feasible allocation on exactly that support.  It never
consults the certified support catalogue or a reference allocation.
"""

from __future__ import annotations

from itertools import product
import re
from typing import Any, Mapping

from .pantry_plan import (
    PantryPlanError,
    PantryPlanValidation,
    parse_pantry_plan_spec,
    validate_pantry_plan,
)


PANTRY_SUPPORT_ACTION_VERSION = "pantry-support-action-dev-v1"
PANTRY_SUPPORT_MASK_TASK = "pantry_support_mask"
PANTRY_SUPPORT_MASK_WIDTH = 6
PANTRY_SUPPORT_MASK_INVALID = "__INVALID_PANTRY_SUPPORT_MASK__"
_MAX_SUPPORT_CHARS = 256


def parse_pantry_support_action(
    candidate: str,
    spec: Mapping[str, Any],
) -> tuple[str, ...]:
    """Parse a support-only action against the prompt-local ingredient IDs."""

    parsed_spec = parse_pantry_plan_spec(spec)
    text = str(candidate).strip()
    if not text or len(text) > _MAX_SUPPORT_CHARS:
        raise PantryPlanError("support action is empty or too long")
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    text = re.sub(r"^(?:support|ingredients?)\s*[:=]\s*", "", text, flags=re.I)
    pieces = tuple(
        piece for piece in re.split(r"[\s,;+]+", text.lower()) if piece
    )
    if not parsed_spec.min_ingredients <= len(pieces) <= parsed_spec.max_ingredients:
        raise PantryPlanError("support action uses the wrong number of ingredients")
    if len(set(pieces)) != len(pieces):
        raise PantryPlanError("support action contains a duplicate ingredient")
    allowed = parsed_spec.ingredient_by_id
    unknown = [piece for piece in pieces if piece not in allowed]
    if unknown:
        raise PantryPlanError(f"unknown ingredient {unknown[0]!r}")
    return tuple(sorted(pieces))


def project_pantry_support(
    support: tuple[str, ...],
    spec: Mapping[str, Any],
) -> PantryPlanValidation | None:
    """Search only the selected support's registered quantity lattice."""

    parsed_spec = parse_pantry_plan_spec(spec)
    by_id = parsed_spec.ingredient_by_id
    if (
        not parsed_spec.min_ingredients <= len(support) <= parsed_spec.max_ingredients
        or len(set(support)) != len(support)
        or any(ingredient_id not in by_id for ingredient_id in support)
    ):
        return None
    grids = []
    for ingredient_id in support:
        ingredient = by_id[ingredient_id]
        grids.append(
            range(
                ingredient.min_if_used_g,
                ingredient.available_g + 1,
                ingredient.step_g,
            )
        )
    for quantities in product(*grids):
        allocation = ";".join(
            f"{ingredient_id}={grams}"
            for ingredient_id, grams in zip(support, quantities)
        )
        validation = validate_pantry_plan(allocation, spec)
        if validation is not None:
            return validation
    return None


def validate_pantry_support_action(
    candidate: str,
    spec: Mapping[str, Any],
) -> PantryPlanValidation | None:
    """Return a verifier-bound mode after deterministic quantity projection."""

    try:
        support = parse_pantry_support_action(candidate, spec)
    except (PantryPlanError, ValueError):
        return None
    return project_pantry_support(support, spec)

def pantry_support_from_mask(
    mask: str,
    spec: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """Decode one fixed-width inclusion mask using the public pantry row order."""

    parsed_spec = parse_pantry_plan_spec(spec)
    text = str(mask).strip()
    if (
        len(parsed_spec.ingredients) != PANTRY_SUPPORT_MASK_WIDTH
        or len(text) != PANTRY_SUPPORT_MASK_WIDTH
        or any(bit not in {"0", "1"} for bit in text)
    ):
        return None
    support = tuple(
        ingredient.ingredient_id
        for bit, ingredient in zip(text, parsed_spec.ingredients)
        if bit == "1"
    )
    if not parsed_spec.min_ingredients <= len(support) <= parsed_spec.max_ingredients:
        return None
    return tuple(sorted(support))


def pantry_support_mask_from_allocation(
    allocation: str,
    spec: Mapping[str, Any],
) -> str:
    """Project a verifier-valid allocation onto its public-row support mask.

    This is the inverse representation adapter needed when an offline replay
    pool stores PantryPlan witnesses but the policy acts on the registered
    support-mask surface. It validates the original witness first and changes
    only its serialization; no replay row is dropped or deduplicated.
    """

    parsed_spec = parse_pantry_plan_spec(spec)
    if len(parsed_spec.ingredients) != PANTRY_SUPPORT_MASK_WIDTH:
        raise PantryPlanError(
            "Pantry support masks require exactly "
            f"{PANTRY_SUPPORT_MASK_WIDTH} public ingredient rows"
        )
    validation = validate_pantry_plan(allocation, spec)
    if validation is None:
        raise PantryPlanError("replayed Pantry allocation fails its verifier")
    selected = {ingredient_id for ingredient_id, _grams in validation.allocations_g}
    mask = "".join(
        "1" if ingredient.ingredient_id in selected else "0"
        for ingredient in parsed_spec.ingredients
    )
    decoded_support = pantry_support_from_mask(mask, spec)
    if decoded_support != tuple(sorted(selected)):
        raise PantryPlanError("Pantry allocation cannot be represented by its mask")
    return mask


def decode_pantry_support_mask(
    mask: str,
    spec: Mapping[str, Any],
) -> str:
    """Project a six-bit policy action into the exact allocation grader surface.

    Every bit string is a syntactically valid fixed-horizon policy action.
    Masks with an invalid support width or no feasible quantity assignment map
    to a fail-closed sentinel that the ordinary PantryPlan verifier rejects.
    """

    support = pantry_support_from_mask(mask, spec)
    if support is None:
        return PANTRY_SUPPORT_MASK_INVALID
    validation = project_pantry_support(support, spec)
    if validation is None:
        return PANTRY_SUPPORT_MASK_INVALID
    return ";".join(
        f"{ingredient_id}={grams}"
        for ingredient_id, grams in validation.allocations_g
    )
