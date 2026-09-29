#!/usr/bin/env python3
"""Materialize PantryPlan with exhaustively certified ingredient supports."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import itertools
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset, DatasetDict


ROOT = Path(__file__).resolve().parents[1]

from modebench.pantry_plan import (  # noqa: E402
    PANTRY_PLAN_VERIFIER,
    PANTRY_PLAN_VERSION,
    parse_pantry_plan_spec,
    validate_pantry_plan,
)


DEFAULT_INGREDIENTS = ROOT / "var/data/pantry_plan_v1/ingredients.json"
DEFAULT_OUTPUT = ROOT / "var/data/pantry_plan_modebench_v1"
FAMILY_POOLS = {
    "plant_protein_bowl": (
        "black_beans_canned",
        "chickpeas_canned",
        "hummus",
        "kale",
        "baby_spinach",
        "broccoli",
        "grape_tomatoes",
        "carrots",
        "pumpkin_seeds",
        "sunflower_seeds",
    ),
    "breakfast_formulation": (
        "rolled_oats",
        "granny_smith_apple",
        "banana",
        "navel_orange",
        "almonds",
        "peanut_butter",
        "pumpkin_seeds",
        "sunflower_seeds",
    ),
    "low_sodium_pantry_meal": (
        "black_beans_canned",
        "chickpeas_canned",
        "kale",
        "baby_spinach",
        "broccoli",
        "grape_tomatoes",
        "carrots",
        "rolled_oats",
        "pumpkin_seeds",
        "sunflower_seeds",
    ),
    "high_fiber_snack": (
        "granny_smith_apple",
        "banana",
        "navel_orange",
        "carrots",
        "grape_tomatoes",
        "almonds",
        "peanut_butter",
        "pumpkin_seeds",
        "sunflower_seeds",
        "hummus",
    ),
}
SODIUM_CAPS = {
    "plant_protein_bowl": Decimal("700"),
    "breakfast_formulation": Decimal("100"),
    "low_sodium_pantry_meal": Decimal("300"),
    "high_fiber_snack": Decimal("200"),
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _decimal_string(value: Decimal) -> str:
    rounded = value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return format(rounded.normalize(), "f")


def _allocation_totals(
    ingredients: list[Mapping[str, Any]],
    allocation: Mapping[str, int],
) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = {"mass_g": Decimal(0)}
    for ingredient in ingredients:
        grams = int(allocation.get(str(ingredient["id"]), 0))
        if not grams:
            continue
        totals["mass_g"] += Decimal(grams)
        for attribute, raw_amount in ingredient["attributes_per_100g"].items():
            totals[str(attribute)] = totals.get(str(attribute), Decimal(0)) + (
                Decimal(str(raw_amount)) * Decimal(grams) / Decimal(100)
            )
    return totals


def enumerate_pantry_supports(
    spec: Mapping[str, Any],
) -> dict[tuple[str, ...], str]:
    """Return one accepted allocation for every exact feasible support."""

    parsed = parse_pantry_plan_spec(spec)
    representatives: dict[tuple[str, ...], str] = {}
    for size in range(parsed.min_ingredients, parsed.max_ingredients + 1):
        for support_rows in itertools.combinations(parsed.ingredients, size):
            if any(row.tags & parsed.forbidden_tags for row in support_rows):
                continue
            quantity_grids = [
                range(row.min_if_used_g, row.available_g + 1, row.step_g)
                for row in support_rows
            ]
            for quantities in itertools.product(*quantity_grids):
                allocation = tuple(
                    sorted(
                        (row.ingredient_id, int(grams))
                        for row, grams in zip(support_rows, quantities)
                    )
                )
                candidate = ";".join(
                    f"{ingredient_id}={grams}"
                    for ingredient_id, grams in allocation
                )
                validation = validate_pantry_plan(candidate, spec)
                if validation is not None:
                    support = tuple(ingredient_id for ingredient_id, _ in allocation)
                    representatives[support] = candidate
                    break
    return representatives


def _ingredient_spec(source_row: Mapping[str, Any], available_g: int) -> dict[str, Any]:
    return {
        "id": str(source_row["id"]),
        "available_g": int(available_g),
        "step_g": 25,
        "min_if_used_g": 50,
        "attributes_per_100g": dict(source_row["attributes_per_100g"]),
        "tags": list(source_row["tags"]),
    }


def _targets_from_seed(
    family: str,
    totals: Mapping[str, Decimal],
) -> dict[str, dict[str, str]] | None:
    sodium_max = min(
        SODIUM_CAPS[family],
        totals["sodium_mg"] * Decimal("1.40") + Decimal("25"),
    )
    if totals["sodium_mg"] > sodium_max:
        return None
    mass_min = max(Decimal("100"), totals["mass_g"] - Decimal("25"))
    mass_max = totals["mass_g"] + Decimal("50")
    return {
        "mass_g": {
            "min": _decimal_string(mass_min),
            "max": _decimal_string(mass_max),
        },
        "energy_kcal": {
            "min": _decimal_string(totals["energy_kcal"] * Decimal("0.72")),
            "max": _decimal_string(totals["energy_kcal"] * Decimal("1.30")),
        },
        "protein_g": {
            "min": _decimal_string(totals["protein_g"] * Decimal("0.62"))
        },
        "fiber_g": {
            "min": _decimal_string(totals["fiber_g"] * Decimal("0.62"))
        },
        "sodium_mg": {"max": _decimal_string(sodium_max)},
    }


def _prompt(family: str, spec: Mapping[str, Any]) -> str:
    family_label = family.replace("_", " ")
    lines = [
        f"Create one feasible {family_label} from this frozen pantry.",
        "Nutrition values are per 100 g. Quantities are exact integer grams.",
        "",
        "Pantry:",
    ]
    for ingredient in spec["ingredients"]:
        attributes = ingredient["attributes_per_100g"]
        tag_text = ",".join(ingredient["tags"]) or "none"
        lines.append(
            "- {id}: available={available_g}g, step={step_g}g, minimum-if-used="
            "{min_if_used_g}g; energy={energy_kcal}kcal, protein={protein_g}g, "
            "fiber={fiber_g}g, sodium={sodium_mg}mg; tags={tag_text}".format(
                **ingredient,
                **attributes,
                tag_text=tag_text,
            )
        )
    lines.extend(
        [
            "",
            f"Use {spec['min_ingredients']} to {spec['max_ingredients']} ingredients.",
            "Forbidden tags: "
            + (", ".join(spec["forbidden_tags"]) or "none"),
            "Targets:",
        ]
    )
    for attribute, bounds in spec["targets"].items():
        rendered = []
        if "min" in bounds:
            rendered.append(f"minimum {bounds['min']}")
        if "max" in bounds:
            rendered.append(f"maximum {bounds['max']}")
        lines.append(f"- {attribute}: {', '.join(rendered)}")
    lines.extend(
        [
            "",
            "Return only ingredient_id=grams pairs separated by semicolons inside "
            "\\boxed{}. Do not add a recipe name or preparation prose.",
        ]
    )
    return "\n".join(lines)


def _build_row(
    *,
    family: str,
    split: str,
    seed: int,
    index: int,
    rng: random.Random,
    source_by_id: Mapping[str, Mapping[str, Any]],
    ingredient_table_sha256: str,
) -> dict[str, Any] | None:
    selected_ids = sorted(rng.sample(list(FAMILY_POOLS[family]), 6))
    ingredients = [
        _ingredient_spec(source_by_id[ingredient_id], rng.choice((100, 125, 150)))
        for ingredient_id in selected_ids
    ]
    available_by_id = {row["id"]: int(row["available_g"]) for row in ingredients}
    possible_forbidden = sorted(
        {
            tag
            for row in ingredients
            for tag in row["tags"]
            if tag in {"peanut", "tree_nut"}
        }
    )
    forbidden = []
    if possible_forbidden and rng.random() < 0.25:
        forbidden = [rng.choice(possible_forbidden)]
    eligible = [
        row
        for row in ingredients
        if not set(row["tags"]) & set(forbidden)
    ]
    if len(eligible) < 4:
        return None
    seed_rows = rng.sample(eligible, rng.choice((2, 3, 3, 4)))
    seed_allocation = {
        row["id"]: rng.choice(
            tuple(range(50, available_by_id[row["id"]] + 1, 25))
        )
        for row in seed_rows
    }
    totals = _allocation_totals(ingredients, seed_allocation)
    targets = _targets_from_seed(family, totals)
    if targets is None:
        return None
    spec: dict[str, Any] = {
        "verifier": PANTRY_PLAN_VERIFIER,
        "pantry_version": PANTRY_PLAN_VERSION,
        "family": family,
        "instance_id": f"{split}-{seed}-{family}-{index}",
        "source_ingredient_table_sha256": ingredient_table_sha256,
        "ingredients": ingredients,
        "targets": targets,
        "min_ingredients": 2,
        "max_ingredients": 4,
        "forbidden_tags": forbidden,
        "certified_mode_count": 2,
    }
    supports = enumerate_pantry_supports(spec)
    if not 8 <= len(supports) <= 64:
        return None
    spec["certified_mode_count"] = len(supports)
    support_keys = ["+".join(support) for support in sorted(supports)]
    spec["certified_support_sha256"] = hashlib.sha256(
        "\n".join(support_keys).encode("utf-8")
    ).hexdigest()
    representative_candidates = list(supports.values())[:2]
    if len(representative_candidates) != 2 or any(
        validate_pantry_plan(candidate, spec) is None
        for candidate in representative_candidates
    ):
        raise RuntimeError("certified PantryPlan representatives failed validation")
    return {
        "problem": _prompt(family, spec),
        "answer": json.dumps(spec, sort_keys=True, separators=(",", ":")),
        "modebench_task": PANTRY_PLAN_VERIFIER,
        "answer_mode_family": family,
        "answer_mode_count": len(supports),
        "answer_mode_split": split,
    }


def _build_rows(
    *,
    per_family: int,
    split: str,
    seed: int,
    curation: Mapping[str, Any],
    excluded_fingerprints: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    source_by_id = {row["id"]: row for row in curation["ingredients"]}
    blocked = excluded_fingerprints or set()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    family_counts: defaultdict[str, int] = defaultdict(int)
    attempts = 0
    while any(family_counts[family] < per_family for family in FAMILY_POOLS):
        attempts += 1
        if attempts > per_family * len(FAMILY_POOLS) * 2_000:
            raise RuntimeError("could not generate the requested PantryPlan rows")
        available_families = [
            family
            for family in FAMILY_POOLS
            if family_counts[family] < per_family
        ]
        family = rng.choice(available_families)
        row = _build_row(
            family=family,
            split=split,
            seed=seed,
            index=family_counts[family],
            rng=rng,
            source_by_id=source_by_id,
            ingredient_table_sha256=str(curation["ingredient_table_sha256"]),
        )
        if row is None:
            continue
        spec = json.loads(row["answer"])
        fingerprint_payload = dict(spec)
        fingerprint_payload.pop("instance_id", None)
        fingerprint = _canonical_sha256(fingerprint_payload)
        if fingerprint in blocked or fingerprint in seen:
            continue
        row["instance_fingerprint"] = fingerprint
        seen.add(fingerprint)
        rows.append(row)
        family_counts[family] += 1
    rng.shuffle(rows)
    return rows


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    return _canonical_sha256(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredients", type=Path, default=DEFAULT_INGREDIENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-per-family", type=int, default=24)
    parser.add_argument("--eval-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=74001)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_per_family < 1 or args.eval_per_family < 1:
        raise ValueError("per-family row counts must be positive")
    curation = json.loads(args.ingredients.read_text(encoding="utf-8"))
    if curation.get("schema_version") != "pantry-plan-ingredient-curation-v1":
        raise ValueError("ingredient curation schema mismatch")
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite")
        if output_root.name != "pantry_plan_modebench_v1":
            raise SystemExit("refusing overwrite outside pantry_plan_modebench_v1")
        shutil.rmtree(output_root)

    train_rows = _build_rows(
        per_family=args.train_per_family,
        split="train",
        seed=args.seed,
        curation=curation,
    )
    train_fingerprints = {row["instance_fingerprint"] for row in train_rows}
    eval_rows = _build_rows(
        per_family=args.eval_per_family,
        split="eval",
        seed=args.seed + 10_000,
        curation=curation,
        excluded_fingerprints=train_fingerprints,
    )
    eval_fingerprints = {row["instance_fingerprint"] for row in eval_rows}
    if train_fingerprints & eval_fingerprints:
        raise RuntimeError("PantryPlan train/eval fingerprints overlap")

    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
        str(output_root / "train")
    )
    DatasetDict({"multi_answer": Dataset.from_list(eval_rows)}).save_to_disk(
        str(output_root / "eval")
    )
    all_rows = train_rows + eval_rows
    family_counts = {
        split: {
            family: sum(row["answer_mode_family"] == family for row in rows)
            for family in FAMILY_POOLS
        }
        for split, rows in (("train", train_rows), ("eval", eval_rows))
    }
    mode_counts = [int(row["answer_mode_count"]) for row in all_rows]
    identity = {
        "schema": "pantry_plan_modebench_v1",
        "seed": args.seed,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "family_counts": family_counts,
        "families": list(FAMILY_POOLS),
        "train_rows_sha256": _rows_sha256(train_rows),
        "eval_rows_sha256": _rows_sha256(eval_rows),
        "ingredient_curation_sha256": _canonical_sha256(curation),
        "ingredient_table_sha256": curation["ingredient_table_sha256"],
        "source_archive_sha256": curation["source"]["archive_sha256"],
        "minimum_exact_support_count": min(mode_counts),
        "maximum_exact_support_count": max(mode_counts),
        "support": "finite_exact_exhaustive",
        "train_eval_overlap_count": 0,
        "manual_source_review_status": curation["status"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[pantry-data] "
        f"train={len(train_rows)} eval={len(eval_rows)} "
        f"modes={min(mode_counts)}..{max(mode_counts)} output={output_root}"
    )


if __name__ == "__main__":
    main()
