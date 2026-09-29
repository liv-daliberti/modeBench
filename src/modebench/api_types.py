"""Stable types for the supported ModeBench public interface."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TypeAlias, TypedDict, Union

JSONValue: TypeAlias = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]
Domain: TypeAlias = Literal[
    "countdown", "graph_coloring", "python_factors", "mathir", "pantry_plan"
]
Split: TypeAlias = Literal["train", "dev", "eval"]
PromptCondition: TypeAlias = Literal["registered_hints_v1", "python_level3_neutral_v1"]
GradeStatus: TypeAlias = Literal[
    "correct",
    "incorrect",
    "malformed",
    "timeout",
    "invalid_reference",
    "worker_failure",
    "resource_limit",
]
EvaluationStatus: TypeAlias = Literal["completed", "failed"]
PathLike: TypeAlias = Union[str, Path]


class ChatMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


class GradeResult(TypedDict):
    status: GradeStatus
    verified: bool
    canonical_key: str | None
    graded_text: str
    detail: str | None


class ModelIdentity(TypedDict):
    id: str
    revision: str


class RunMetadata(TypedDict):
    model: ModelIdentity
    generation: dict[str, JSONValue]
    prompt_condition: PromptCondition


@dataclass(frozen=True)
class Task:
    id: str
    level: int
    domain: Domain
    problem: str
    answer: Mapping[str, JSONValue]
    dataset_sha256: str | None = None


@dataclass(frozen=True)
class EvaluationOptions:
    min_defined_prompts: int = 30
    run: RunMetadata | None = None
    data_root: PathLike | None = None
    config: str | None = None
    split: Split = "eval"
    allow_partial: bool = False


@dataclass(frozen=True)
class EvaluationReceipt:
    output: Path
    status: EvaluationStatus
    prompts: int
    cells: int
    work_dir: Path


@dataclass(frozen=True)
class DiversitySummary:
    d_mode: float | None
    reportable: bool
    defined_prompts: int | None
    prompts: int | None
    standard_error: float | None
    reason: str | None = None


@dataclass(frozen=True)
class CellSummary:
    level: int
    domain: Domain
    k: int
    accuracy: float | None
    pass_at_k: float | None
    distinct_at_k: float | None
    pcmd: DiversitySummary
    protocol_notes: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    schema: str
    status: EvaluationStatus
    cells: tuple[CellSummary, ...]
    input_sha256: str | None


class Progress(TypedDict):
    phase: str
    completed_prompts: int
    total_prompts: int | None


class Draw(TypedDict):
    attempts: list[GradeResult]


class PromptResult(TypedDict):
    id: str
    reference_sha256: str
    prompt_sha256: str | None
    prompt_profile: dict[str, JSONValue] | None
    draws: list[Draw]
    pass_at_k: float | None
    distinct_at_k: int | None
    accuracy: float | None
