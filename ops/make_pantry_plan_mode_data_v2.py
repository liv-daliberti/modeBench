#!/usr/bin/env python3
"""Materialize the prospective three-way PantryPlan v2 split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from datasets import Dataset, DatasetDict


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1]))
OPS = Path(os.environ.get("OAT_ZERO_OPS_ROOT", ROOT / "ops"))
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_pantry_plan_mode_data import (  # noqa: E402
    FAMILY_POOLS,
    _build_rows,
    _canonical_sha256,
    _rows_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ingredients",
        type=Path,
        default=ROOT / "var/data/pantry_plan_v1/ingredients.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "var/data/pantry_plan_modebench_v2",
    )
    parser.add_argument("--train-per-family", type=int, default=96)
    parser.add_argument("--dev-per-family", type=int, default=16)
    parser.add_argument("--eval-per-family", type=int, default=32)
    parser.add_argument("--seed", type=int, default=74002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.train_per_family, args.dev_per_family, args.eval_per_family) < 1:
        raise ValueError("all three per-family row counts must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh PantryPlan v2 root required: {output_root}")
    curation = json.loads(args.ingredients.read_text(encoding="utf-8"))
    if curation.get("schema_version") != "pantry-plan-ingredient-curation-v1":
        raise ValueError("ingredient curation schema mismatch")

    train_rows = _build_rows(
        per_family=args.train_per_family,
        split="train",
        seed=args.seed,
        curation=curation,
    )
    train_fingerprints = {row["instance_fingerprint"] for row in train_rows}
    dev_rows = _build_rows(
        per_family=args.dev_per_family,
        split="dev",
        seed=args.seed + 10_000,
        curation=curation,
        excluded_fingerprints=train_fingerprints,
    )
    dev_fingerprints = {row["instance_fingerprint"] for row in dev_rows}
    eval_rows = _build_rows(
        per_family=args.eval_per_family,
        split="eval",
        seed=args.seed + 20_000,
        curation=curation,
        excluded_fingerprints=train_fingerprints | dev_fingerprints,
    )
    eval_fingerprints = {row["instance_fingerprint"] for row in eval_rows}
    if (
        train_fingerprints & dev_fingerprints
        or train_fingerprints & eval_fingerprints
        or dev_fingerprints & eval_fingerprints
    ):
        raise RuntimeError("PantryPlan v2 split fingerprints overlap")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
            str(staging / "train")
        )
        DatasetDict({"multi_answer": Dataset.from_list(dev_rows)}).save_to_disk(
            str(staging / "dev")
        )
        DatasetDict({"multi_answer": Dataset.from_list(eval_rows)}).save_to_disk(
            str(staging / "eval")
        )
        rows_by_split = {
            "train": train_rows,
            "dev": dev_rows,
            "eval": eval_rows,
        }
        all_rows = train_rows + dev_rows + eval_rows
        mode_counts = [int(row["answer_mode_count"]) for row in all_rows]
        identity = {
            "schema": "pantry_plan_modebench_v2",
            "seed": args.seed,
            "split_seeds": {
                "train": args.seed,
                "dev": args.seed + 10_000,
                "eval": args.seed + 20_000,
            },
            "split_rows": {
                split: len(rows) for split, rows in rows_by_split.items()
            },
            "split_rows_sha256": {
                split: _rows_sha256(rows)
                for split, rows in rows_by_split.items()
            },
            "family_counts": {
                split: {
                    family: sum(
                        row["answer_mode_family"] == family for row in rows
                    )
                    for family in FAMILY_POOLS
                }
                for split, rows in rows_by_split.items()
            },
            "families": list(FAMILY_POOLS),
            "ingredient_curation_sha256": _canonical_sha256(curation),
            "ingredient_table_sha256": curation["ingredient_table_sha256"],
            "source_archive_sha256": curation["source"]["archive_sha256"],
            "minimum_exact_support_count": min(mode_counts),
            "maximum_exact_support_count": max(mode_counts),
            "support": "finite_exact_exhaustive",
            "split_overlap_count": 0,
            "v1_rows_copied": 0,
            "manual_source_review_status": curation["status"],
        }
        (staging / "identity.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        "[pantry-data-v2] "
        f"train={len(train_rows)} dev={len(dev_rows)} eval={len(eval_rows)} "
        f"modes={min(mode_counts)}..{max(mode_counts)} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
