from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from modebench.grading import (
    PYTHON_FACTOR_RESPONSE_SURFACE_VERSION,
    boxed_reward_fn,
    extract_normalized_final_answer,
    validated_modebench_exploration_identity,
    validated_modebench_outcome_key,
)
from modebench.python_modebench import (
    PYTHON_FACTOR_VERIFIER,
    PYTHON_FACTOR_VERSION,
    parse_python_factor_candidate,
    python_factor_mode_count,
)
from modebench.python_modebench_process import PythonFactorVerifierProcess
from ops.make_python_factor_mode_data import _build_rows, _certified_programs


ROOT = Path(__file__).resolve().parents[1]


def _spec() -> dict:
    return {
        "verifier": PYTHON_FACTOR_VERIFIER,
        "python_version": PYTHON_FACTOR_VERSION,
        "cases": [6, 10, 15],
        "source": "test",
    }


def test_external_python_tool_certifies_distinct_behavior_modes():
    grader = PythonFactorVerifierProcess(timeout_seconds=2)
    try:
        first = grader.validate("lambda n: 2 if n % 2 == 0 else 3", _spec())
        second = grader.validate("lambda n: 3 if n % 3 == 0 else 5", _spec())

        assert first is not None
        assert second is not None
        assert first.outputs == (2, 2, 3)
        assert second.outputs == (3, 5, 3)
        assert first.canonical_key != second.canonical_key
        assert grader._process is not None
        assert grader._process.pid != 0
    finally:
        grader.close()


def test_python_factor_grader_binds_reward_and_key_to_same_tool_calls():
    reference = json.dumps(_spec())
    response = r"\boxed{lambda n: 2 if n % 2 == 0 else 3}"

    info, reward = boxed_reward_fn(response, reference)
    key = validated_modebench_outcome_key(response, reference)
    extracted = extract_normalized_final_answer(
        response,
        template="qwen_boxed",
        gt_answer=reference,
    )

    assert info == {"formatted": True}
    assert reward == 1.0
    assert key == "python_factor:2,2,3"
    assert extracted == key
    identity = validated_modebench_exploration_identity(response, reference)
    assert identity is not None
    assert identity.endpoint_key == key
    assert identity.route_signature is not None
    assert "python-factor-route:v1:" in identity.route_signature
    assert "2" not in identity.route_signature
    assert "3" not in identity.route_signature


def test_python_factor_grader_normalizes_boxed_latex_lambda_surface():
    assert (
        PYTHON_FACTOR_RESPONSE_SURFACE_VERSION
        == "python-factor-response-v2-latex-lambda"
    )
    reference = json.dumps(_spec())
    expected_key = "python_factor:2,2,3"

    for response in (
        r"\boxed{\lambda n: 2 if n % 2 == 0 else 3}",
        r"\boxed{\lambda\,n: 2 if n % 2 == 0 else 3}",
    ):
        info, reward = boxed_reward_fn(response, reference)
        assert info == {"formatted": True}
        assert reward == 1.0
        assert validated_modebench_outcome_key(response, reference) == expected_key

    # The alias is formatting-only: it cannot change the required signature
    # or bypass the existing restricted-AST safety boundary.
    assert (
        validated_modebench_outcome_key(r"\boxed{\lambda x: 2}", reference)
        is None
    )
    assert (
        validated_modebench_outcome_key(
            r"\boxed{\lambda n: __import__('os').system('id')}",
            reference,
        )
        is None
    )


def test_python_factor_grader_rejects_wrong_and_non_integer_outputs():
    reference = json.dumps(_spec())

    _, wrong_reward = boxed_reward_fn(r"\boxed{lambda n: 4}", reference)
    _, bool_reward = boxed_reward_fn(r"\boxed{lambda n: n == n}", reference)

    assert wrong_reward == 0.0
    assert bool_reward == 0.0
    assert validated_modebench_outcome_key("lambda n: 4", reference) is None


def test_python_factor_syntax_rejects_calls_import_surfaces_and_other_names():
    for candidate in (
        "lambda n: __import__('os').system('id')",
        "lambda n: open('/tmp/x')",
        "lambda x: 2",
        "lambda n: [2][0]",
    ):
        try:
            parse_python_factor_candidate(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe candidate was accepted: {candidate}")


def test_generator_certifies_exact_multiple_modes_and_disjoint_behavior_seeds():
    rows = _build_rows(
        3,
        seed=5100,
        split_tag="test",
        case_count=4,
        max_value=48,
        min_modes=16,
    )

    assert len(rows) == 3
    for row in rows:
        spec = json.loads(row["answer"])
        cases = tuple(spec["cases"])
        assert row["modebench_task"] == PYTHON_FACTOR_VERIFIER
        assert row["answer_mode_count"] == python_factor_mode_count(cases)
        assert row["answer_mode_count"] >= 16
        assert spec["num_externally_certified_modes"] == 2
        assert len(set(_certified_programs(cases))) == 2


def test_external_verifier_cli_reports_the_executed_mode():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops/verify_python_factor_mode.py"),
            "--candidate",
            "lambda n: 2 if n % 2 == 0 else 3",
            "--reference",
            json.dumps(_spec()),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )

    payload = json.loads(completed.stdout)
    assert payload == {
        "canonical_key": "python_factor:2,2,3",
        "outputs": [2, 2, 3],
        "valid": True,
    }
