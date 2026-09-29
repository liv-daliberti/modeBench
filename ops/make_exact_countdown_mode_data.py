#!/usr/bin/env python3
"""Build an exact multi-answer Countdown benchmark for answer-mode coverage."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from itertools import combinations
from pathlib import Path

from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]

from make_modebench_data import (  # noqa: E402
    _countdown_expression_map,
    _countdown_prompt,
)
from modebench.grading import (  # noqa: E402
    _canonical_countdown_expression_key,
)


def _canonical_expression_keys(
    numbers: list[int],
    target: int,
    expressions: set[str] | None = None,
) -> set[str]:
    spec = {"verifier": "countdown", "numbers": numbers, "target": target}
    keys: set[str] = set()
    if expressions is None:
        expressions = _countdown_expression_map(numbers).get(int(target), set())
    for expression in expressions:
        key = _canonical_countdown_expression_key(expression, spec)
        if key is not None:
            keys.add(key)
    return keys


def _synthetic_countdown_rows(
    limit: int,
    *,
    seed: int,
    split_tag: str,
    number_count: int,
    max_value: int,
    min_modes: int,
    max_modes: int | None,
    exclude: set[tuple[tuple[int, ...], int]] | None = None,
) -> list[dict[str, str | int]]:
    """Sample `limit` problems; `exclude` holds (numbers, target) identities of
    previously drawn splits so evaluation prompts are genuinely held out —
    train and eval otherwise sample from the same exhaustive pool."""
    rng = random.Random(seed)
    number_count = max(3, min(int(number_count), 4))
    number_upper = max(6, min(int(max_value), 14))
    candidates: list[tuple[list[int], int, set[str]]] = []
    for numbers_tuple in combinations(range(2, number_upper + 1), number_count):
        numbers = list(numbers_tuple)
        expressions_by_target = _countdown_expression_map(numbers)
        for target, expressions in expressions_by_target.items():
            if target <= 0 or target in numbers:
                continue
            if abs(target) > number_upper * max(number_count, 1):
                continue
            if not expressions:
                continue
            if exclude and (tuple(numbers), int(target)) in exclude:
                continue
            keys = _canonical_expression_keys(numbers, target, expressions)
            if len(keys) < int(min_modes):
                continue
            if max_modes is not None and len(keys) > int(max_modes):
                continue
            candidates.append((numbers, int(target), keys))
    if len(candidates) < int(limit):
        raise RuntimeError(
            f"Only built {len(candidates)} {split_tag} candidates; requested {limit}. "
            "Relax Countdown mode-count bounds."
        )
    rng.shuffle(candidates)

    rows: list[dict[str, str | int]] = []
    for numbers, target, keys in candidates[: int(limit)]:
        spec = {
            "verifier": "countdown",
            "numbers": numbers,
            "target": target,
            "source": "synthetic_exact_countdown",
            "instance_id": f"{split_tag}-{seed}-{len(rows)}",
            "num_completions": len(keys),
            "num_expressions": len(keys),
        }
        rows.append(
            {
                "problem": _countdown_prompt(numbers, target),
                "answer": json.dumps(spec, sort_keys=True),
                "modebench_task": "countdown",
                "answer_mode_count": len(keys),
                "answer_mode_split": split_tag,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--eval-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number-count", type=int, default=4)
    parser.add_argument("--max-value", type=int, default=12)
    parser.add_argument("--multi-min-modes", type=int, default=4)
    parser.add_argument("--multi-max-modes", type=int, default=32)
    parser.add_argument("--unique-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_size <= 0:
        raise SystemExit("--train-size must be positive")
    if args.eval_size <= 0:
        raise SystemExit("--eval-size must be positive")

    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    common = {
        "number_count": args.number_count,
        "max_value": args.max_value,
    }
    train_rows = _synthetic_countdown_rows(
        args.train_size,
        seed=args.seed,
        split_tag="train_multi_answer",
        min_modes=args.multi_min_modes,
        max_modes=args.multi_max_modes,
        **common,
    )
    train_identities = {
        (tuple(json.loads(row["answer"])["numbers"]), int(json.loads(row["answer"])["target"]))
        for row in train_rows
    }
    eval_multi_rows = _synthetic_countdown_rows(
        args.eval_size,
        seed=args.seed + 10_000,
        split_tag="eval_multi_answer",
        min_modes=args.multi_min_modes,
        max_modes=args.multi_max_modes,
        exclude=train_identities,
        **common,
    )

    eval_splits = {"multi_answer": Dataset.from_list(eval_multi_rows)}
    if args.unique_eval:
        eval_unique_rows = _synthetic_countdown_rows(
            args.eval_size,
            seed=args.seed + 20_000,
            split_tag="eval_unique_answer",
            min_modes=1,
            max_modes=1,
            exclude=train_identities,
            **common,
        )
        eval_splits["unique_answer"] = Dataset.from_list(eval_unique_rows)

    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
        str(output_root / "train")
    )
    DatasetDict(eval_splits).save_to_disk(str(output_root / "eval"))
    print(
        "wrote train={} eval={} rows={}/{}".format(
            output_root / "train",
            output_root / "eval",
            len(train_rows),
            len(eval_multi_rows),
        )
    )


if __name__ == "__main__":
    main()
