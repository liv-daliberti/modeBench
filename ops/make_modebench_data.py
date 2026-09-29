#!/usr/bin/env python3
"""Build a local graph-coloring + Countdown verifier benchmark."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download


GRAPH_DATASET = "brozonoyer/gram-graph-coloring"
COUNTDOWN_DATASET = "Chenmien/Countdown"


def _graph_edges(flat_input: list[int], n: int) -> list[list[int]]:
    values = [int(value) for value in flat_input]
    edge_marker = 2 if 2 in values else 1
    edges: list[list[int]] = []
    index = 0
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if index >= len(values):
                raise ValueError(f"Graph input too short for n={n}.")
            if values[index] == edge_marker:
                edges.append([u, v])
            index += 1
    return edges


def _graph_color_string(colors: list[int | None]) -> str:
    return "".join("?" if color is None else str(int(color)) for color in colors)


def _valid_graph_colorings(n: int, edges: list[list[int]]) -> list[list[int]]:
    valid: list[list[int]] = []
    for value in range(3**n):
        colors = []
        current = value
        for _ in range(n):
            colors.append((current % 3) + 1)
            current //= 3
        if all(colors[u - 1] != colors[v - 1] for u, v in edges):
            valid.append(colors)
    return valid


def _normalize_graph_target(target: Any) -> list[int]:
    raw = [int(value) for value in target]
    unique = sorted(set(raw))
    mapping = {value: index + 1 for index, value in enumerate(unique)}
    return [mapping[value] for value in raw]


def _choose_partial_graph_coloring(
    *,
    n: int,
    edges: list[list[int]],
    target: Any,
    rng: random.Random,
    hidden_count: int,
    min_completions: int,
) -> tuple[list[int | None], int] | None:
    target_colors = _normalize_graph_target(target)
    valid_colorings = _valid_graph_colorings(n, edges)
    if len(valid_colorings) < min_completions:
        return None
    hidden_count = max(1, min(hidden_count, n))
    indices = list(range(n))
    for _ in range(32):
        hidden = set(rng.sample(indices, hidden_count))
        partial: list[int | None] = [
            None if index in hidden else target_colors[index] for index in indices
        ]
        completion_count = sum(
            all(
                partial[index] is None or colors[index] == partial[index]
                for index in indices
            )
            for colors in valid_colorings
        )
        if completion_count >= min_completions:
            return partial, completion_count
    return None


def _graph_prompt(n: int, edges: list[list[int]], partial: list[int | None]) -> str:
    edge_text = ", ".join(f"({u},{v})" for u, v in edges) if edges else "none"
    partial_text = _graph_color_string(partial)
    hidden_count = sum(color is None for color in partial)
    return (
        f"Color this graph with vertices 1 through {n}. The undirected edges are: "
        f"{edge_text}. Assign each vertex one color from {{1, 2, 3}} so that no "
        "edge has the same color on both endpoints. A partial coloring is "
        f"{partial_text}, where ? means uncolored. Keep every shown digit fixed. "
        f"There are exactly {hidden_count} question marks. Return exactly one "
        f"final answer and no explanation: exactly {hidden_count} digits, each "
        "one 1, 2, or 3, for the missing positions from left to right, inside "
        "\\boxed{}."
    )


def _hub_graph_rows(
    split: str,
    limit: int,
    *,
    seed: int,
    max_n: int,
    max_edges: int,
    min_solutions: int,
    hidden_count: int,
    min_completions: int,
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    rows = _load_parquet_rows_from_hub(GRAPH_DATASET, split)
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples: list[dict[str, str]] = []
    for row in rows:
        n = int(row["n"])
        edges = _graph_edges(list(row["input"]), n)
        num_solutions = int(row.get("num_solutions", 0))
        if n > max_n:
            continue
        if len(edges) > max_edges:
            continue
        if num_solutions < min_solutions:
            continue
        partial_bundle = _choose_partial_graph_coloring(
            n=n,
            edges=edges,
            target=row["target"],
            rng=rng,
            hidden_count=hidden_count,
            min_completions=min_completions,
        )
        if partial_bundle is None:
            continue
        partial, completion_count = partial_bundle
        spec = {
            "verifier": "graph_coloring",
            "n": n,
            "edges": edges,
            "partial_colors": partial,
            "source": GRAPH_DATASET,
            "instance_id": str(row.get("instance_id", "")),
            "num_solutions": num_solutions,
            "num_completions": completion_count,
        }
        examples.append(
            {
                "problem": _graph_prompt(n, edges, partial),
                "answer": json.dumps(spec, sort_keys=True),
                "modebench_task": "graph_coloring",
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _synthetic_graph_rows(
    limit: int,
    *,
    seed: int,
    max_n: int,
    max_edges: int,
    min_solutions: int,
    hidden_count: int,
    min_completions: int,
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    rng = random.Random(seed)
    examples: list[dict[str, str]] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...], str]] = set()
    max_n = max(4, min(max_n, 6))
    max_possible_attempts = max(2_000, limit * 200)
    for _ in range(max_possible_attempts):
        n = rng.randint(4, max_n)
        all_edges = [(u, v) for u in range(1, n + 1) for v in range(u + 1, n + 1)]
        edge_upper = min(max_edges, len(all_edges), n + 1)
        edge_lower = min(edge_upper, max(1, n - 2))
        edge_count = rng.randint(edge_lower, edge_upper)
        edges = [list(edge) for edge in sorted(rng.sample(all_edges, edge_count))]
        valid_colorings = _valid_graph_colorings(n, edges)
        if len(valid_colorings) < min_solutions:
            continue
        target = rng.choice(valid_colorings)
        partial_bundle = _choose_partial_graph_coloring(
            n=n,
            edges=edges,
            target=target,
            rng=rng,
            hidden_count=hidden_count,
            min_completions=min_completions,
        )
        if partial_bundle is None:
            continue
        partial, completion_count = partial_bundle
        key = (n, tuple(tuple(edge) for edge in edges), _graph_color_string(partial))
        if key in seen:
            continue
        seen.add(key)
        spec = {
            "verifier": "graph_coloring",
            "n": n,
            "edges": edges,
            "partial_colors": partial,
            "source": "synthetic_graph_coloring",
            "instance_id": f"synthetic-{seed}-{len(examples)}",
            "num_solutions": len(valid_colorings),
            "num_completions": completion_count,
        }
        examples.append(
            {
                "problem": _graph_prompt(n, edges, partial),
                "answer": json.dumps(spec, sort_keys=True),
                "modebench_task": "graph_coloring",
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _graph_rows(
    split: str,
    limit: int,
    *,
    seed: int,
    source: str,
    max_n: int,
    max_edges: int,
    min_solutions: int,
    hidden_count: int,
    min_completions: int,
) -> list[dict[str, str]]:
    if source == "synthetic":
        return _synthetic_graph_rows(
            limit,
            seed=seed,
            max_n=max_n,
            max_edges=max_edges,
            min_solutions=min_solutions,
            hidden_count=hidden_count,
            min_completions=min_completions,
        )
    return _hub_graph_rows(
        split,
        limit,
        seed=seed,
        max_n=max_n,
        max_edges=max_edges,
        min_solutions=min_solutions,
        hidden_count=hidden_count,
        min_completions=min_completions,
    )


def _load_parquet_rows_from_hub(dataset_name: str, split: str) -> list[dict[str, Any]]:
    import pandas as pd

    api = HfApi()
    prefix = f"data/{split}-"
    parquet_files = [
        path
        for path in api.list_repo_files(dataset_name, repo_type="dataset")
        if path.startswith(prefix) and path.endswith(".parquet")
    ]
    if not parquet_files:
        raise ValueError(f"No parquet files found for {dataset_name} split {split}.")
    rows: list[dict[str, Any]] = []
    for parquet_file in sorted(parquet_files):
        local_path = hf_hub_download(
            dataset_name,
            parquet_file,
            repo_type="dataset",
        )
        rows.extend(pd.read_parquet(local_path).to_dict("records"))
    return rows


COUNTDOWN_RE = re.compile(
    r"Using the numbers \[(?P<numbers>[0-9,\s-]+)\], create an equation that equals "
    r"(?P<target>-?\d+)",
    flags=re.IGNORECASE,
)


def _countdown_problem_text(row: dict[str, Any]) -> str | None:
    for key in ("prompt", "text", "question", "input", "problem"):
        value = row.get(key)
        if isinstance(value, str) and "Using the numbers" in value:
            return value
    for value in row.values():
        if isinstance(value, str) and "Using the numbers" in value:
            return value
    return None


def _parse_countdown_row(row: dict[str, Any]) -> tuple[list[int], int] | None:
    if "numbers" in row and "target" in row:
        try:
            return [int(value) for value in row["numbers"]], int(row["target"])
        except Exception:
            pass
    text = _countdown_problem_text(row)
    if text is None:
        return None
    match = COUNTDOWN_RE.search(text)
    if match is None:
        return None
    numbers = [int(value.strip()) for value in match.group("numbers").split(",")]
    return numbers, int(match.group("target"))


def _countdown_prompt(numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers {numbers}, create an arithmetic expression that equals "
        f"{target}. Use each given number exactly once. You may use +, -, *, /, "
        "and parentheses. Return exactly one final answer and no explanation: the "
        "expression inside \\boxed{}."
    )


def _countdown_expression_map(numbers: list[int]) -> dict[int, set[str]]:
    by_value: dict[Fraction, set[str]] = {}

    def build(items: tuple[tuple[Fraction, str], ...]) -> dict[Fraction, set[str]]:
        if len(items) == 1:
            value, expr = items[0]
            return {value: {expr}}
        values: dict[Fraction, set[str]] = {}
        for split_at in range(1, len(items)):
            left_values = build(items[:split_at])
            right_values = build(items[split_at:])
            for left_value, left_exprs in left_values.items():
                for right_value, right_exprs in right_values.items():
                    for left_expr in left_exprs:
                        for right_expr in right_exprs:
                            candidates = [
                                (left_value + right_value, f"({left_expr}+{right_expr})"),
                                (left_value - right_value, f"({left_expr}-{right_expr})"),
                                (left_value * right_value, f"({left_expr}*{right_expr})"),
                            ]
                            if right_value != 0:
                                candidates.append(
                                    (
                                        left_value / right_value,
                                        f"({left_expr}/{right_expr})",
                                    )
                                )
                            for value, expr in candidates:
                                values.setdefault(value, set()).add(expr)
        return values

    for order in set(permutations(numbers)):
        items = tuple((Fraction(value), str(value)) for value in order)
        for value, expressions in build(items).items():
            by_value.setdefault(value, set()).update(expressions)
    return {
        int(value): expressions
        for value, expressions in by_value.items()
        if value.denominator == 1
    }


def _first_split(dataset_dict) -> str:
    if "train" in dataset_dict:
        return "train"
    return next(iter(dataset_dict.keys()))


def _hub_countdown_rows(
    limit: int,
    *,
    seed: int,
    max_numbers: int,
    max_value: int,
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    dataset_dict = load_dataset(COUNTDOWN_DATASET)
    dataset = dataset_dict[_first_split(dataset_dict)]
    rows = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples: list[dict[str, str]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for row in rows:
        parsed = _parse_countdown_row(dict(row))
        if parsed is None:
            continue
        numbers, target = parsed
        if len(numbers) > max_numbers:
            continue
        if max(abs(value) for value in numbers) > max_value:
            continue
        if abs(target) > max_value * max(len(numbers), 1):
            continue
        key = (tuple(numbers), target)
        if key in seen:
            continue
        seen.add(key)
        spec = {
            "verifier": "countdown",
            "numbers": numbers,
            "target": target,
            "source": COUNTDOWN_DATASET,
        }
        examples.append(
            {
                "problem": _countdown_prompt(numbers, target),
                "answer": json.dumps(spec, sort_keys=True),
                "modebench_task": "countdown",
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _synthetic_countdown_rows(
    limit: int,
    *,
    seed: int,
    max_numbers: int,
    max_value: int,
    min_expressions: int,
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    rng = random.Random(seed)
    examples: list[dict[str, str]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    num_count = max(3, min(max_numbers, 4))
    number_upper = max(5, min(max_value, 12))
    max_possible_attempts = max(5_000, limit * 500)
    for _ in range(max_possible_attempts):
        numbers = sorted(rng.sample(range(2, number_upper + 1), num_count))
        expressions_by_target = _countdown_expression_map(numbers)
        viable_targets = [
            target
            for target, expressions in expressions_by_target.items()
            if target > 0
            and target not in numbers
            and abs(target) <= max_value * max(num_count, 1)
            and len(expressions) >= min_expressions
        ]
        if not viable_targets:
            continue
        target = rng.choice(viable_targets)
        key = (tuple(numbers), target)
        if key in seen:
            continue
        seen.add(key)
        spec = {
            "verifier": "countdown",
            "numbers": numbers,
            "target": target,
            "source": "synthetic_countdown",
            "num_expressions": len(expressions_by_target[target]),
        }
        examples.append(
            {
                "problem": _countdown_prompt(numbers, target),
                "answer": json.dumps(spec, sort_keys=True),
                "modebench_task": "countdown",
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _countdown_rows(
    limit: int,
    *,
    seed: int,
    source: str,
    max_numbers: int,
    max_value: int,
    min_expressions: int,
) -> list[dict[str, str]]:
    if source == "synthetic":
        return _synthetic_countdown_rows(
            limit,
            seed=seed,
            max_numbers=max_numbers,
            max_value=max_value,
            min_expressions=min_expressions,
        )
    return _hub_countdown_rows(
        limit,
        seed=seed,
        max_numbers=max_numbers,
        max_value=max_value,
    )


def _split_counts(total: int, graph_frac: float) -> tuple[int, int]:
    graph_count = int(round(float(total) * float(graph_frac)))
    graph_count = max(0, min(int(total), graph_count))
    return graph_count, int(total) - graph_count


def _build_rows(
    size: int,
    *,
    graph_frac: float,
    seed: int,
    graph_split: str,
    graph_source: str,
    graph_max_n: int,
    graph_max_edges: int,
    graph_min_solutions: int,
    graph_hidden_count: int,
    graph_min_completions: int,
    countdown_source: str,
    countdown_max_numbers: int,
    countdown_max_value: int,
    countdown_min_expressions: int,
) -> list[dict[str, str]]:
    graph_count, countdown_count = _split_counts(size, graph_frac)
    graph = _graph_rows(
        graph_split,
        graph_count,
        seed=seed,
        source=graph_source,
        max_n=graph_max_n,
        max_edges=graph_max_edges,
        min_solutions=graph_min_solutions,
        hidden_count=graph_hidden_count,
        min_completions=graph_min_completions,
    )
    countdown = _countdown_rows(
        countdown_count,
        seed=seed + 10_000,
        source=countdown_source,
        max_numbers=countdown_max_numbers,
        max_value=countdown_max_value,
        min_expressions=countdown_min_expressions,
    )
    rows = graph + countdown
    if len(graph) < graph_count:
        raise RuntimeError(
            f"Only found {len(graph)} graph-coloring rows, requested {graph_count}. "
            "Relax graph filters or lower --graph-frac."
        )
    if len(countdown) < countdown_count:
        raise RuntimeError(
            f"Only found {len(countdown)} Countdown rows, requested {countdown_count}. "
            "Relax Countdown filters or lower --graph-frac."
        )
    random.Random(seed + 20_000).shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph-frac", type=float, default=0.75)
    parser.add_argument("--graph-source", choices=("synthetic", "hub"), default="synthetic")
    parser.add_argument("--graph-train-split", default="train")
    parser.add_argument("--graph-eval-split", default="test")
    parser.add_argument("--graph-max-n", type=int, default=5)
    parser.add_argument("--graph-max-edges", type=int, default=5)
    parser.add_argument("--graph-min-solutions", type=int, default=12)
    parser.add_argument("--graph-hidden-count", type=int, default=2)
    parser.add_argument("--graph-min-completions", type=int, default=2)
    parser.add_argument(
        "--countdown-source",
        choices=("synthetic", "hub"),
        default="synthetic",
    )
    parser.add_argument("--countdown-max-numbers", type=int, default=3)
    parser.add_argument("--countdown-max-value", type=int, default=12)
    parser.add_argument("--countdown-min-expressions", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_size <= 0:
        raise SystemExit("--train-size must be positive")
    if args.eval_size <= 0:
        raise SystemExit("--eval-size must be positive")
    if not 0.0 <= args.graph_frac <= 1.0:
        raise SystemExit("--graph-frac must be between 0 and 1")

    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    train_rows = _build_rows(
        args.train_size,
        graph_frac=args.graph_frac,
        seed=args.seed,
        graph_split=args.graph_train_split,
        graph_source=args.graph_source,
        graph_max_n=args.graph_max_n,
        graph_max_edges=args.graph_max_edges,
        graph_min_solutions=args.graph_min_solutions,
        graph_hidden_count=args.graph_hidden_count,
        graph_min_completions=args.graph_min_completions,
        countdown_source=args.countdown_source,
        countdown_max_numbers=args.countdown_max_numbers,
        countdown_max_value=args.countdown_max_value,
        countdown_min_expressions=args.countdown_min_expressions,
    )
    eval_rows = _build_rows(
        args.eval_size,
        graph_frac=args.graph_frac,
        seed=args.seed + 100_000,
        graph_split=args.graph_eval_split,
        graph_source=args.graph_source,
        graph_max_n=args.graph_max_n,
        graph_max_edges=args.graph_max_edges,
        graph_min_solutions=args.graph_min_solutions,
        graph_hidden_count=args.graph_hidden_count,
        graph_min_completions=args.graph_min_completions,
        countdown_source=args.countdown_source,
        countdown_max_numbers=args.countdown_max_numbers,
        countdown_max_value=args.countdown_max_value,
        countdown_min_expressions=args.countdown_min_expressions,
    )

    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
        str(output_root / "train")
    )
    DatasetDict({"modebench": Dataset.from_list(eval_rows)}).save_to_disk(
        str(output_root / "eval")
    )
    print(
        "wrote train={} eval={} rows={}/{}".format(
            output_root / "train",
            output_root / "eval",
            len(train_rows),
            len(eval_rows),
        )
    )


if __name__ == "__main__":
    main()
