"""Incremental quality gate: new public APIs and the resilient evaluation path."""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED = [
    "src/modebench/api.py",
    "src/modebench/api_types.py",
    "src/modebench/contracts.py",
    "src/modebench/streaming.py",
    "src/modebench/parallel.py",
    "src/modebench/reporting.py",
    "src/modebench/evaluation.py",
    "src/modebench/cli.py",
    "ops/benchmark_evaluation.py",
    "ops/summarize_levels.py",
    "ops/check_quality.py",
    "tests/test_streaming.py",
    "tests/test_public_api.py",
    "tests/typecheck/public_api.py",
]


def main():
    commands = [
        ["ruff", "check", *CHECKED],
        ["ruff", "format", "--check", *CHECKED],
        [
            "mypy",
            "src/modebench/api.py",
            "src/modebench/api_types.py",
            "src/modebench/contracts.py",
            "tests/typecheck/public_api.py",
        ],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-m", *command], cwd=ROOT, check=True)
    subprocess.run(
        [str(Path(sys.executable).parent / "cffconvert"), "--validate"], cwd=ROOT, check=True
    )
    # If annotations degrade to Any, these mistakes would silently type-check.
    with tempfile.TemporaryDirectory(prefix="modebench-type-contract-") as temporary:
        invalid = Path(temporary) / "invalid.py"
        invalid.write_text(
            'from modebench.api import grade, EvaluationOptions\ngrade("not a Task", 123)\nEvaluationOptions(min_defined_prompts="many")\n'
        )
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--no-incremental", str(invalid)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 1 or result.stdout.count("[arg-type]") != 3:
            raise AssertionError(
                "public API negative type contract failed: " + result.stdout + result.stderr
            )


if __name__ == "__main__":
    main()
