#!/usr/bin/env python3
"""Materialize the executable Python-factor ModeBench environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modebench.python_modebench import (  # noqa: E402
    PYTHON_FACTOR_VERIFIER,
    PYTHON_FACTOR_VERSION,
    proper_divisors,
    python_factor_mode_count,
)
from modebench.python_modebench_process import (  # noqa: E402
    validate_python_factor_function_external,
)


def _mapping_lambda(cases: tuple[int, ...], outputs: tuple[int, ...]) -> str:
    branches = [
        f"{output} if n=={case} else "
        for case, output in zip(cases[:-1], outputs[:-1])
    ]
    return "lambda n:" + "".join(branches) + str(outputs[-1])


def _certified_programs(cases: tuple[int, ...]) -> tuple[str, str]:
    divisor_sets = [proper_divisors(value) for value in cases]
    smallest = tuple(divisors[0] for divisors in divisor_sets)
    largest = tuple(divisors[-1] for divisors in divisor_sets)
    return (
        _mapping_lambda(cases, smallest),
        _mapping_lambda(cases, largest),
    )


def _prompt(cases: tuple[int, ...]) -> str:
    rendered = ", ".join(str(value) for value in cases)
    return (
        "Write one pure Python function with the exact form lambda n: EXPR. "
        f"An external Python tool will call it once for each n in [{rendered}]. "
        "For every call, return an integer d satisfying 1 < d < n and n % d == 0. "
        "Different valid return vectors are different solution modes. You may use "
        "integer literals, n, +, -, *, //, %, comparisons, Boolean operators, and "
        "conditional expressions; calls, imports, attributes, containers, and "
        "other names are forbidden. Return exactly the one-line lambda inside "
        "\\boxed{} and no explanation."
    )


def _row(
    *,
    cases: tuple[int, ...],
    split_tag: str,
    seed: int,
    index: int,
) -> dict[str, Any]:
    mode_count = python_factor_mode_count(cases)
    spec: dict[str, Any] = {
        "verifier": PYTHON_FACTOR_VERIFIER,
        "python_version": PYTHON_FACTOR_VERSION,
        "cases": list(cases),
        "source": "synthetic_python_factor_modebench_v1",
        "instance_id": f"{split_tag}-{seed}-{index}",
        "num_modes": mode_count,
    }
    validations = [
        validate_python_factor_function_external(program, spec)
        for program in _certified_programs(cases)
    ]
    if any(validation is None for validation in validations):
        raise RuntimeError("external Python certification rejected a seed program")
    keys = {validation.canonical_key for validation in validations if validation}
    if len(keys) < 2:
        raise RuntimeError("row does not retain two externally certified modes")
    spec["num_externally_certified_modes"] = len(keys)
    spec["certified_mode_key_sha256"] = hashlib.sha256(
        "\n".join(sorted(keys)).encode("utf-8")
    ).hexdigest()
    return {
        "problem": _prompt(cases),
        "answer": json.dumps(spec, sort_keys=True, separators=(",", ":")),
        "modebench_task": PYTHON_FACTOR_VERIFIER,
        "answer_mode_count": mode_count,
        "answer_mode_split": split_tag,
    }


def _candidate_values(max_value: int) -> list[int]:
    return [
        value
        for value in range(6, int(max_value) + 1)
        if len(proper_divisors(value)) >= 2
    ]


def _build_rows(
    count: int,
    *,
    seed: int,
    split_tag: str,
    case_count: int,
    max_value: int,
    min_modes: int,
    excluded: set[tuple[int, ...]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    values = _candidate_values(max_value)
    if len(values) < case_count:
        raise RuntimeError("bounded value range cannot supply enough cases")
    blocked = excluded or set()
    seen: set[tuple[int, ...]] = set()
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < int(count):
        attempts += 1
        if attempts > max(20_000, int(count) * 500):
            raise RuntimeError(f"could not build {count} unique Python-factor rows")
        cases = tuple(sorted(rng.sample(values, case_count)))
        if cases in seen or cases in blocked:
            continue
        if python_factor_mode_count(cases) < min_modes:
            continue
        rows.append(
            _row(
                cases=cases,
                split_tag=split_tag,
                seed=seed,
                index=len(rows),
            )
        )
        seen.add(cases)
    rng.shuffle(rows)
    return rows


def _row_cases(rows: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    return {
        tuple(int(value) for value in json.loads(row["answer"])["cases"])
        for row in rows
    }


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=384)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument("--max-value", type=int, default=96)
    parser.add_argument("--min-modes", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_size <= 0 or args.eval_size <= 0:
        raise SystemExit("train and eval sizes must be positive")
    if not 2 <= args.case_count <= 8:
        raise SystemExit("--case-count must be between 2 and 8")
    if args.min_modes < 2:
        raise SystemExit("--min-modes must be at least 2")

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite")
        if output_root.name != "python_factor_modebench_v1":
            raise SystemExit(
                "refusing overwrite outside the exact python_factor_modebench_v1 target"
            )
        shutil.rmtree(output_root)

    train_rows = _build_rows(
        args.train_size,
        seed=args.seed,
        split_tag="train",
        case_count=args.case_count,
        max_value=args.max_value,
        min_modes=args.min_modes,
    )
    eval_rows = _build_rows(
        args.eval_size,
        seed=args.seed + 10_000,
        split_tag="eval",
        case_count=args.case_count,
        max_value=args.max_value,
        min_modes=args.min_modes,
        excluded=_row_cases(train_rows),
    )
    if _row_cases(train_rows) & _row_cases(eval_rows):
        raise RuntimeError("train/eval Python-factor identity overlap")

    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
        str(output_root / "train")
    )
    DatasetDict({"multi_answer": Dataset.from_list(eval_rows)}).save_to_disk(
        str(output_root / "eval")
    )
    mode_counts = [
        int(row["answer_mode_count"]) for row in train_rows + eval_rows
    ]
    identity = {
        "schema": "python_factor_modebench_v1",
        "seed": int(args.seed),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_rows_sha256": _rows_sha256(train_rows),
        "eval_rows_sha256": _rows_sha256(eval_rows),
        "case_count": int(args.case_count),
        "max_value": int(args.max_value),
        "minimum_exact_mode_count": min(mode_counts),
        "maximum_exact_mode_count": max(mode_counts),
        "external_verifier": "isolated_python_jsonl_worker",
        "support": "finite_exact",
    }
    (output_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote Python-factor train={len(train_rows)} eval={len(eval_rows)} "
        f"to {output_root}"
    )
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
