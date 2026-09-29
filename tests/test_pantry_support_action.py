from __future__ import annotations

import json

from pathlib import Path

from modebench.data import load_split

from modebench.pantry_support_action import (
    parse_pantry_support_action,
    project_pantry_support,
    validate_pantry_support_action,
)


def _first_dev_spec():
    row = load_split(
        Path(__file__).resolve().parents[1] / "data",
        "level1_pantry_plan",
        "dev",
    )[0]
    return json.loads(row["answer"])


def test_support_projection_finds_a_real_allocation_without_catalogue():
    spec = _first_dev_spec()
    support = ("pumpkin_seeds", "rolled_oats")
    result = project_pantry_support(support, spec)
    assert result is not None
    assert result.canonical_key.endswith("pumpkin_seeds+rolled_oats")
    assert tuple(name for name, _grams in result.allocations_g) == support


def test_support_parser_accepts_only_prompt_local_unique_ids():
    spec = _first_dev_spec()
    assert parse_pantry_support_action(
        "SUPPORT: rolled_oats + pumpkin_seeds", spec
    ) == ("pumpkin_seeds", "rolled_oats")
    assert validate_pantry_support_action("rolled_oats rolled_oats", spec) is None
    assert validate_pantry_support_action("rolled_oats gold_dust", spec) is None


def test_projection_does_not_change_the_selected_support():
    spec = _first_dev_spec()
    result = validate_pantry_support_action(
        "black_beans_canned pumpkin_seeds rolled_oats", spec
    )
    if result is not None:
        assert tuple(name for name, _grams in result.allocations_g) == (
            "black_beans_canned",
            "pumpkin_seeds",
            "rolled_oats",
        )
