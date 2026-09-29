"""Shared public task identities, independent of historical adapter modules."""

from .api_types import Domain

DOMAINS: tuple[Domain, ...] = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
BENCHMARK_CONTRACT = "modebench-verifiers-v1"


def normalize_task_identity(level: int | str, domain: str) -> tuple[int, Domain]:
    """Normalize registered levels and the retained 'pantry' domain alias."""
    if isinstance(level, str) and level.startswith("level"):
        level = level[5:]
    if isinstance(level, bool) or str(level) not in {"1", "2", "3", "4", "5"}:
        raise ValueError("level must be 1, 2, 3, 4, or 5")
    domain = "pantry_plan" if domain == "pantry" else domain
    if domain not in DOMAINS:
        raise ValueError(f"unsupported ModeBench domain: {domain}")
    return int(level), domain
