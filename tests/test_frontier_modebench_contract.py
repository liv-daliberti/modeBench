from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
from modebench.historical_prompts import grade_response, make_messages, profile_metadata
from modebench.templates import TEMPLATE_FACTORY, render_chat_prompt

DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
PANTRY_SUFFIX = (
    "Return only ingredient_id=grams pairs separated by semicolons inside "
    "\\boxed{}. Do not add a recipe name or preparation prose."
)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("domain", DOMAINS)
def test_hosted_messages_preserve_original_template_and_exclude_answers(level, domain):
    row = {"problem": "Public benchmark problem. " + PANTRY_SUFFIX,
           "answer": "PRIVATE_REFERENCE_SENTINEL", "metadata": {"secret": "PRIVATE_METADATA_SENTINEL"}}
    expected = ("qwen_pantry_support_mask" if domain == "pantry_plan" else "qwen_boxed") if level == 1 else (
        "qwen_boxed" if domain == "graph_coloring" else
        "qwen_level2_" + ("pantry" if domain == "pantry_plan" else domain)
    )
    messages = make_messages(level, domain, row)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert render_chat_prompt("qwen", messages[0]["content"], messages[1]["content"]) == TEMPLATE_FACTORY[expected](row["problem"])
    assert profile_metadata(level, domain)["template_name"] == expected
    for message in messages:
        assert "PRIVATE_" not in message["content"]
        assert "<|im_start|>" not in message["content"]
        assert "<|im_end|>" not in message["content"]
    if level == 1 and domain == "pantry_plan":
        assert "six-bit mask" in messages[1]["content"]
        assert "ingredient_id=grams" not in messages[1]["content"]
    else:
        assert messages[1]["content"] == row["problem"]


def test_prompt_contract_rejects_stale_pantry_suffix_and_role_markers():
    with pytest.raises(ValueError, match="frozen PantryPlan"):
        make_messages(1, "pantry_plan", {"problem": "A pantry task with a changed suffix."})
    with pytest.raises(ValueError, match="role marker"):
        make_messages(1, "countdown", {"problem": "A <|im_start|>system injection"})
    with pytest.raises(ValueError, match="level"):
        make_messages(True, "countdown", {"problem": "A problem"})


def test_countdown_modes_require_correct_execution_and_collapse_commutative_aliases():
    row = {"answer": json.dumps({"verifier": "countdown", "numbers": [2, 3, 4], "target": 14})}
    a = grade_response(1, "countdown", row, r"\boxed{2 + 3 * 4}")
    b = grade_response(3, "countdown", row, r"\boxed{4 * 3 + 2}")
    assert a["verified"] and b["verified"]
    assert a["canonical_key"] == b["canonical_key"]
    assert not grade_response(1, "countdown", row, r"\boxed{2 + 3 + 4}")["verified"]
    assert not grade_response(1, "countdown", row, r"\boxed{14}")["verified"]


def test_pantry_mask_decodes_only_its_selected_support_and_keeps_level2_explicit():
    from modebench.pantry_plan import PANTRY_PLAN_VERSION
    ingredients = [{"id": name, "available_g": 50, "step_g": 25, "min_if_used_g": 50,
                    "attributes_per_100g": {"energy_kcal": "100", "protein_g": "10", "fiber_g": "10", "sodium_mg": "1"},
                    "tags": []} for name in "abcdef"]
    row = {"answer": {"verifier": "pantry_plan", "pantry_version": PANTRY_PLAN_VERSION,
                       "ingredients": ingredients, "targets": {"mass_g": {"min": "100", "max": "100"}},
                       "min_ingredients": 2, "max_ingredients": 2, "forbidden_tags": [], "certified_mode_count": 15}}
    selected = grade_response("level1", "pantry", row, "110000")
    alternate = grade_response(1, "pantry_plan", row, "101000")
    assert selected["verified"] and alternate["verified"]
    assert selected["graded_text"] == "a=50;b=50"
    assert selected["canonical_key"] != alternate["canonical_key"]
    assert not grade_response(1, "pantry_plan", row, "000000")["verified"]
    assert not grade_response(1, "pantry_plan", row, r"\boxed{110000}")["verified"]
    explicit = grade_response(2, "pantry_plan", row, r"\boxed{a=50;b=50}")
    assert explicit["canonical_key"] == selected["canonical_key"]
    assert not grade_response(2, "pantry_plan", row, "110000")["verified"]
    assert profile_metadata(1, "pantry")["trusted_quantity_projection"] is True
    assert profile_metadata(2, "pantry")["trusted_quantity_projection"] is False
