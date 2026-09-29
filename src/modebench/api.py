"""Supported typed API: load tasks, prompt, grade, evaluate files, and read results.

Private helpers and historical adapter modules are not part of this contract.
The older module-level imports remain available for compatibility.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator, cast

from . import data, dataset_cache, prompts, reporting, streaming, verifier
from .api_types import (
    CellSummary,
    ChatMessage,
    DiversitySummary,
    Domain,
    EvaluationOptions,
    EvaluationReceipt,
    EvaluationStatus,
    GradeResult,
    JSONValue,
    PathLike,
    Progress,
    PromptCondition,
    PromptResult,
    Report,
    Split,
    Task,
)
from .diagnostics import VerifierExecutionError
from .validation import InputError, loads

__all__ = [
    "InputError",
    "VerifierExecutionError",
    "Task",
    "EvaluationOptions",
    "EvaluationReceipt",
    "GradeResult",
    "Report",
    "load_tasks",
    "make_prompt",
    "grade",
    "evaluate_file",
    "read_report",
    "iter_results",
]


def load_tasks(
    config: str,
    split: Split = "eval",
    *,
    data_root: PathLike | None = None,
    cache_dir: PathLike | None = None,
    offline: bool = False,
) -> Iterator[Task]:
    """Yield authenticated frozen tasks. References belong only in the evaluator."""
    root = (
        Path(data_root)
        if data_root is not None
        else dataset_cache.fetch(config, split, cache_dir=cache_dir, offline=offline)
    )
    record = data.split_record(root, config, split, frozen=True)
    for index, row in enumerate(data.load_split(root, config, split, frozen=True)):
        answer = loads(row["answer"]) if isinstance(row["answer"], str) else row["answer"]
        yield Task(
            id=data.prompt_id(config, split, index),
            level=record["level"],
            domain=cast(Domain, record["domain"]),
            problem=row["problem"],
            answer=cast(dict[str, JSONValue], answer),
            dataset_sha256=record["parquet_sha256"],
        )


def make_prompt(
    task: Task, condition: PromptCondition = "registered_hints_v1"
) -> list[ChatMessage]:
    """Render registered messages without including the task reference."""
    return cast(
        list[ChatMessage],
        prompts.make_messages(
            task.level,
            task.domain,
            {"problem": task.problem, "answer": dict(task.answer)},
            condition,
        ),
    )


def grade(task: Task, response: str) -> GradeResult:
    """Return structured correctness or evaluator-failure diagnostics."""
    return cast(
        GradeResult,
        verifier.grade_response(task.level, task.domain, {"answer": dict(task.answer)}, response),
    )


def evaluate_file(
    input_path: PathLike,
    output_path: PathLike,
    *,
    options: EvaluationOptions | None = None,
    work_dir: PathLike | None = None,
    resume: bool = False,
    workers: int = 1,
    progress: Callable[[Progress], None] | None = None,
) -> EvaluationReceipt:
    """Evaluate a saved JSONL snapshot with bounded memory and durable checkpoints."""
    result = streaming.evaluate_file(
        input_path,
        output_path,
        **asdict(options or EvaluationOptions()),
        work_dir=work_dir,
        resume=resume,
        workers=workers,
        progress=progress,
    )
    return EvaluationReceipt(
        output=Path(result["output"]),
        status=cast(EvaluationStatus, result["status"]),
        prompts=result["prompts"],
        cells=result["cells"],
        work_dir=Path(result["work_dir"]),
    )


def read_report(path: PathLike) -> Report:
    """Read aggregate scores and verify their detailed-result journal's hash."""
    payload = reporting.read_report(path)
    cells = []
    for cell in payload["cells"]:
        pcmd = cell["pcmd"]
        cells.append(
            CellSummary(
                level=cell["level"],
                domain=cast(Domain, cell["domain"]),
                k=cell["k"],
                accuracy=cell["accuracy"],
                pass_at_k=cell["pass_at_k"],
                distinct_at_k=cell["distinct_at_k"],
                pcmd=DiversitySummary(
                    d_mode=pcmd["d_mode"],
                    reportable=pcmd["reportable"],
                    defined_prompts=pcmd.get("defined_prompts"),
                    prompts=pcmd.get("prompts"),
                    standard_error=pcmd.get("standard_error"),
                    reason=pcmd.get("reason"),
                ),
                protocol_notes=tuple(cell.get("protocol_notes", [])),
            )
        )
    return Report(
        schema=payload["schema"],
        status=cast(EvaluationStatus, payload["evaluation"]["status"]),
        cells=tuple(cells),
        input_sha256=payload.get("input_sha256"),
    )


def iter_results(path: PathLike) -> Iterator[PromptResult]:
    """Stream per-prompt details after verifying the complete journal identity."""
    for result in reporting.iter_prompt_results(path):
        yield cast(PromptResult, result)
