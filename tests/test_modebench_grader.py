from __future__ import annotations

import json

from modebench.grading import (
    boxed_reward_fn,
    extract_normalized_final_answer,
    validated_modebench_exploration_identity,
    validated_modebench_outcome_key,
)


def test_graph_coloring_verifier_accepts_valid_boxed_coloring():
    spec = {
        "verifier": "graph_coloring",
        "n": 4,
        "edges": [[1, 2], [2, 3], [3, 4]],
    }

    info, reward = boxed_reward_fn("work \\boxed{1212}", json.dumps(spec))

    assert info == {"formatted": True}
    assert reward == 1.0


def test_graph_coloring_verifier_accepts_unboxed_compact_coloring():
    spec = {
        "verifier": "graph_coloring",
        "n": 4,
        "edges": [[1, 2], [2, 3], [3, 4]],
    }

    info, reward = boxed_reward_fn("1212", json.dumps(spec))

    assert info == {"formatted": True}
    assert reward == 1.0


def test_graph_coloring_verifier_enforces_partial_colors():
    spec = {
        "verifier": "graph_coloring",
        "n": 4,
        "edges": [[1, 2], [2, 3], [3, 4]],
        "partial_colors": [1, None, None, 2],
    }

    _, valid_reward = boxed_reward_fn("1212", json.dumps(spec))
    _, wrong_partial_reward = boxed_reward_fn("2121", json.dumps(spec))

    assert valid_reward == 1.0
    assert wrong_partial_reward == 0.0


def test_graph_coloring_verifier_accepts_missing_digit_fill():
    spec = {
        "verifier": "graph_coloring",
        "n": 4,
        "edges": [[1, 2], [2, 3], [3, 4]],
        "partial_colors": [1, None, None, 2],
    }

    info, reward = boxed_reward_fn("work \\boxed{21}", json.dumps(spec))
    _, wrong_fill_reward = boxed_reward_fn("work \\boxed{12}", json.dumps(spec))

    assert info == {"formatted": True}
    assert reward == 1.0
    assert wrong_fill_reward == 0.0


def test_graph_coloring_verifier_accepts_canonical_three_digit_action():
    spec = {
        "verifier": "graph_coloring",
        "n": 6,
        "edges": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
        "partial_colors": [1, None, 1, None, 1, None],
    }
    reference = json.dumps(spec)

    info, reward = boxed_reward_fn("222", reference)
    _, invalid_reward = boxed_reward_fn("111", reference)
    mode_key = extract_normalized_final_answer(
        "222", template="qwen_graph_digits", gt_answer=reference
    )

    assert info == {"formatted": True}
    assert reward == 1.0
    assert invalid_reward == 0.0
    assert mode_key == "graph_coloring:121212"


def test_graph_coloring_mode_key_canonicalizes_missing_digit_fill():
    spec = {
        "verifier": "graph_coloring",
        "n": 4,
        "edges": [[1, 2], [2, 3], [3, 4]],
        "partial_colors": [1, None, None, 2],
    }
    gt_answer = json.dumps(spec)

    fill_key = extract_normalized_final_answer(
        "work \\boxed{21}",
        template="qwen_boxed",
        gt_answer=gt_answer,
    )
    full_key = extract_normalized_final_answer(
        "\\boxed{1212}",
        template="qwen_boxed",
        gt_answer=gt_answer,
    )
    other_key = extract_normalized_final_answer(
        "\\boxed{31}",
        template="qwen_boxed",
        gt_answer=gt_answer,
    )

    assert fill_key == "graph_coloring:1212"
    assert full_key == fill_key
    assert other_key == "graph_coloring:1312"


def test_graph_coloring_verifier_rejects_adjacent_same_color():
    spec = {
        "verifier": "graph_coloring",
        "n": 4,
        "edges": [[1, 2], [2, 3], [3, 4]],
    }

    _, reward = boxed_reward_fn("work \\boxed{1123}", json.dumps(spec))

    assert reward == 0.0


def test_countdown_verifier_accepts_valid_expression_with_all_numbers_once():
    spec = {
        "verifier": "countdown",
        "numbers": [2, 3, 4],
        "target": 14,
    }

    info, reward = boxed_reward_fn("work \\boxed{2 + 3 * 4}", json.dumps(spec))

    assert info == {"formatted": True}
    assert reward == 1.0


def test_countdown_verifier_accepts_unboxed_compact_expression():
    spec = {
        "verifier": "countdown",
        "numbers": [2, 3, 4],
        "target": 14,
    }

    info, reward = boxed_reward_fn("2 + 3 * 4", json.dumps(spec))

    assert info == {"formatted": True}
    assert reward == 1.0


def test_countdown_verifier_rejects_expression_that_reuses_number():
    spec = {
        "verifier": "countdown",
        "numbers": [2, 3, 4],
        "target": 10,
    }

    _, reward = boxed_reward_fn("work \\boxed{2 + 4 + 4}", json.dumps(spec))

    assert reward == 0.0


def test_countdown_mode_key_canonicalizes_expression_ast():
    spec = {
        "verifier": "countdown",
        "numbers": [2, 3, 4],
        "target": 14,
    }
    gt_answer = json.dumps(spec)

    left_key = extract_normalized_final_answer(
        "\\boxed{2 + 3 * 4}",
        template="qwen_boxed",
        gt_answer=gt_answer,
    )
    right_key = extract_normalized_final_answer(
        "\\boxed{(4 * 3) + 2}",
        template="qwen_boxed",
        gt_answer=gt_answer,
    )
    wrong_numbers_key = extract_normalized_final_answer(
        "\\boxed{2 + 4 + 4}",
        template="qwen_boxed",
        gt_answer=gt_answer,
    )

    assert left_key == "countdown:add(2,mul(3,4))"
    assert right_key == left_key
    assert wrong_numbers_key is None


def test_countdown_hierarchical_identity_separates_endpoint_and_route():
    first_spec = {
        "verifier": "countdown",
        "numbers": [2, 3, 4],
        "target": 14,
    }
    second_spec = {
        "verifier": "countdown",
        "numbers": [5, 6, 7],
        "target": 47,
    }

    first = validated_modebench_exploration_identity(
        "\\boxed{2 + 3 * 4}",
        json.dumps(first_spec),
    )
    second = validated_modebench_exploration_identity(
        "\\boxed{5 + 6 * 7}",
        json.dumps(second_spec),
    )

    assert first is not None and second is not None
    assert first.endpoint_key != second.endpoint_key
    assert first.route_signature == second.route_signature
    assert first.route_signature == "countdown-route:v1:add(input,mul(input,input))"


def test_validated_modebench_key_binds_execution_and_canonicalization():
    spec = {
        "verifier": "countdown",
        "numbers": [2, 3, 4],
        "target": 14,
    }
    reference = json.dumps(spec)

    assert validated_modebench_outcome_key(
        "\\boxed{2 + 3 * 4}", reference
    ) == "countdown:add(2,mul(3,4))"
    assert validated_modebench_outcome_key(
        "\\boxed{(4 * 3) + 2}", reference
    ) == "countdown:add(2,mul(3,4))"
    assert validated_modebench_outcome_key(
        "\\boxed{2 + 4 + 4}", reference
    ) is None
    assert validated_modebench_outcome_key(
        "\\boxed{2 + 3 + 4}", reference
    ) is None
    # The first equality side uses the right operands but misses the target.
    # Admission must key the exact second AST that actually verified.
    assert validated_modebench_outcome_key(
        "\\boxed{2 + 3 + 4 = 2 + 3 * 4}", reference
    ) == "countdown:add(2,mul(3,4))"


def test_validated_modebench_key_rejects_ordinary_math_answer():
    assert validated_modebench_outcome_key("\\boxed{3}", "3") is None
