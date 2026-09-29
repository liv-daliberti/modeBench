#!/usr/bin/env python3
"""Materialize finite, executable MathIR action-menu ModeBench data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from modebench.mathir import (  # noqa: E402
    MATHIR_MENU_VERIFIER,
    MATHIR_MENU_VERSION,
    enumerate_mathir_action_menu_keys,
    validate_mathir_action_menu,
)


@dataclass(frozen=True)
class Family:
    name: str
    initial_lhs: str
    initial_rhs: str
    display_equation: str
    commands: tuple[str, ...]
    certified_routes: tuple[tuple[str, ...], ...]


FAMILIES = (
    Family(
        name="ax_plus_b_eq_c",
        initial_lhs="add(mul(a,x),b)",
        initial_rhs="c",
        display_equation="a*x + b = c",
        commands=(
            "sub(b)",
            "div(a)",
            "sub(div(b,a))",
            "add(b)",
            "mul(a)",
            "sub(c)",
        ),
        certified_routes=(
            ("sub(b)", "div(a)"),
            ("div(a)", "sub(div(b,a))"),
        ),
    ),
    Family(
        name="x_over_a_plus_b_eq_c",
        initial_lhs="add(div(x,a),b)",
        initial_rhs="c",
        display_equation="x/a + b = c",
        commands=(
            "sub(b)",
            "mul(a)",
            "sub(mul(a,b))",
            "add(b)",
            "div(a)",
            "sub(c)",
        ),
        certified_routes=(
            ("sub(b)", "mul(a)"),
            ("mul(a)", "sub(mul(a,b))"),
        ),
    ),
    Family(
        name="ax_plus_b_eq_dx_plus_c",
        initial_lhs="add(mul(a,x),b)",
        initial_rhs="add(mul(d,x),c)",
        display_equation="a*x + b = d*x + c",
        commands=(
            "sub(b)",
            "sub(mul(d,x))",
            "div(sub(a,d))",
            "sub(add(mul(d,x),b))",
            "add(b)",
            "div(a)",
        ),
        certified_routes=(
            ("sub(b)", "sub(mul(d,x))", "div(sub(a,d))"),
            ("sub(mul(d,x))", "sub(b)", "div(sub(a,d))"),
            ("sub(add(mul(d,x),b))", "div(sub(a,d))"),
        ),
    ),
    Family(
        name="ax_plus_b_eq_c_minus_dx",
        initial_lhs="add(mul(a,x),b)",
        initial_rhs="sub(c,mul(d,x))",
        display_equation="a*x + b = c - d*x",
        commands=(
            "sub(b)",
            "add(mul(d,x))",
            "div(add(a,d))",
            "sub(sub(b,mul(d,x)))",
            "add(b)",
            "div(a)",
        ),
        certified_routes=(
            ("sub(b)", "add(mul(d,x))", "div(add(a,d))"),
            ("add(mul(d,x))", "sub(b)", "div(add(a,d))"),
            ("sub(sub(b,mul(d,x)))", "div(add(a,d))"),
        ),
    ),
)


def _sample_bindings(family: Family, rng: random.Random) -> dict[str, int]:
    nonzero = [value for value in range(-9, 10) if value != 0]
    offsets = list(range(-12, 13))
    solution = rng.choice(nonzero)
    a = rng.choice(nonzero)
    b = rng.choice(offsets)
    if family.name == "ax_plus_b_eq_c":
        return {"a": a, "b": b, "c": a * solution + b}
    if family.name == "x_over_a_plus_b_eq_c":
        quotient = solution
        return {"a": a, "b": b, "c": quotient + b}
    d_candidates = [
        value
        for value in nonzero
        if (
            family.name == "ax_plus_b_eq_dx_plus_c"
            and value != a
        )
        or (
            family.name == "ax_plus_b_eq_c_minus_dx"
            and value != -a
        )
    ]
    d = rng.choice(d_candidates)
    if family.name == "ax_plus_b_eq_dx_plus_c":
        return {"a": a, "b": b, "c": (a - d) * solution + b, "d": d}
    if family.name == "ax_plus_b_eq_c_minus_dx":
        return {"a": a, "b": b, "c": (a + d) * solution + b, "d": d}
    raise RuntimeError(f"unknown MathIR action-menu family: {family.name}")


def _menu_for_row(
    family: Family,
    *,
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, str]]:
    shuffled = list(family.commands)
    rng.shuffle(shuffled)
    actions = {
        chr(ord("A") + index): command
        for index, command in enumerate(shuffled)
    }
    command_to_action = {command: action for action, command in actions.items()}
    return actions, command_to_action


def _reference_spec(
    *,
    family: Family,
    bindings: dict[str, int],
    actions: dict[str, str],
    split_tag: str,
    seed: int,
    index: int,
) -> dict[str, Any]:
    return {
        "verifier": MATHIR_MENU_VERIFIER,
        "mathir_version": MATHIR_MENU_VERSION,
        "bindings": bindings,
        "initial_lhs": family.initial_lhs,
        "initial_rhs": family.initial_rhs,
        "max_steps": 4,
        "actions": actions,
        "support_is_open": False,
        "source": "synthetic_mathir_action_menu_v1",
        "family": family.name,
        "instance_id": f"{split_tag}-{seed}-{index}",
    }


def _prompt(
    family: Family,
    bindings: dict[str, int],
    actions: dict[str, str],
) -> str:
    binding_text = ", ".join(f"{name}={bindings[name]}" for name in sorted(bindings))
    menu = "; ".join(f"{action} means {command}" for action, command in actions.items())
    return (
        "MathIR linear-menu-v1. Solve for x by choosing an executable action "
        "sequence. Every selected action is applied exactly to BOTH sides of "
        "the current equation, followed by exact simplification. "
        f"Bindings: {binding_text}. Initial equation: {family.display_equation}. "
        f"Action menu: {menu}. "
        "Choose 1-4 action IDs. Illegal operations, repeated states, and paths "
        "that do not finish with x isolated are rejected. Return only the IDs "
        "separated by semicolons inside the answer box; for example formatting "
        "only: \\boxed{A;C}. Do not return x's numeric value."
    )


def _row(
    *,
    family: Family,
    bindings: dict[str, int],
    split_tag: str,
    seed: int,
    index: int,
    rng: random.Random,
    family_support: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    actions, command_to_action = _menu_for_row(family, rng=rng)
    spec = _reference_spec(
        family=family,
        bindings=bindings,
        actions=actions,
        split_tag=split_tag,
        seed=seed,
        index=index,
    )
    certified_keys = set()
    for route in family.certified_routes:
        action_program = ";".join(command_to_action[command] for command in route)
        validation = validate_mathir_action_menu(action_program, spec)
        if validation is None:
            raise RuntimeError(
                f"{family.name} certified action route failed: {action_program}"
            )
        certified_keys.add(validation.canonical_key)
    if len(certified_keys) < 2:
        raise RuntimeError(f"{family.name} lacks two distinct certified routes")
    mode_count, support_digest = family_support[family.name]
    spec["num_completions"] = mode_count
    spec["valid_mode_count"] = mode_count
    spec["valid_mode_key_sha256"] = support_digest
    return {
        "problem": _prompt(family, bindings, actions),
        "answer": json.dumps(spec, sort_keys=True, separators=(",", ":")),
        "modebench_task": MATHIR_MENU_VERIFIER,
        "answer_mode_count": mode_count,
        "answer_mode_split": split_tag,
        "mathir_family": family.name,
    }


def _family_support() -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for family_index, family in enumerate(FAMILIES):
        rng = random.Random(590_000 + family_index)
        bindings = _sample_bindings(family, rng)
        actions = {
            chr(ord("A") + index): command
            for index, command in enumerate(family.commands)
        }
        spec = _reference_spec(
            family=family,
            bindings=bindings,
            actions=actions,
            split_tag="support_audit",
            seed=590_000,
            index=family_index,
        )
        keys = enumerate_mathir_action_menu_keys(spec)
        if len(keys) < 2:
            raise RuntimeError(f"{family.name} has fewer than two valid modes")
        digest = hashlib.sha256(
            "\n".join(sorted(keys)).encode("utf-8")
        ).hexdigest()
        result[family.name] = (len(keys), digest)
    return result


def _build_rows(
    count: int,
    *,
    seed: int,
    split_tag: str,
    family_support: dict[str, tuple[int, str]],
    excluded: set[tuple[str, tuple[tuple[str, int], ...]]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    excluded = excluded or set()
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < int(count):
        attempts += 1
        if attempts > max(20_000, int(count) * 300):
            raise RuntimeError(f"could not build {count} unique MathIR rows")
        family = FAMILIES[len(rows) % len(FAMILIES)]
        bindings = _sample_bindings(family, rng)
        identity = (family.name, tuple(sorted(bindings.items())))
        if identity in excluded or identity in seen:
            continue
        rows.append(
            _row(
                family=family,
                bindings=bindings,
                split_tag=split_tag,
                seed=seed,
                index=len(rows),
                rng=rng,
                family_support=family_support,
            )
        )
        seen.add(identity)
    rng.shuffle(rows)
    return rows


def _identities(
    rows: list[dict[str, Any]],
) -> set[tuple[str, tuple[tuple[str, int], ...]]]:
    return {
        (
            str(spec["family"]),
            tuple(sorted((str(key), int(value)) for key, value in spec["bindings"].items())),
        )
        for row in rows
        for spec in (json.loads(row["answer"]),)
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
    parser.add_argument("--seed", type=int, default=5900)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_size <= 0 or args.eval_size <= 0:
        raise SystemExit("train and eval sizes must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite")
        if output_root.name != "mathir_action_menu_v1":
            raise SystemExit(
                "refusing overwrite outside the exact mathir_action_menu_v1 target"
            )
        shutil.rmtree(output_root)

    family_support = _family_support()
    train_rows = _build_rows(
        args.train_size,
        seed=args.seed,
        split_tag="train",
        family_support=family_support,
    )
    eval_rows = _build_rows(
        args.eval_size,
        seed=args.seed + 10_000,
        split_tag="multi_answer",
        family_support=family_support,
        excluded=_identities(train_rows),
    )
    if _identities(train_rows) & _identities(eval_rows):
        raise RuntimeError("train/eval MathIR identity overlap")

    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
        str(output_root / "train")
    )
    DatasetDict({"multi_answer": Dataset.from_list(eval_rows)}).save_to_disk(
        str(output_root / "eval")
    )
    mode_counts = sorted({count for count, _digest in family_support.values()})
    identity = {
        "schema": "mathir_action_menu_v1",
        "seed": int(args.seed),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_rows_sha256": _rows_sha256(train_rows),
        "eval_rows_sha256": _rows_sha256(eval_rows),
        "families": [family.name for family in FAMILIES],
        "support": "finite_exhaustively_enumerated",
        "mode_counts": mode_counts,
        "family_support": {
            family: {"mode_count": count, "mode_key_sha256": digest}
            for family, (count, digest) in sorted(family_support.items())
        },
        "training_bank_initialization": "empty",
        "gold_mode_catalogue_supplied_to_policy": False,
        "canonical_identity": "normalized_states_from_same_successful_execution",
    }
    (output_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote MathIR action-menu train={len(train_rows)} "
        f"eval={len(eval_rows)} to {output_root}"
    )
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
