"""Read compact receipts and stream hash-bound detailed results."""

import re
from pathlib import Path
from typing import Any, Iterator

from .contracts import normalize_task_identity
from .streaming import file_sha256
from .validation import InputError, loads, positive_integer

SCHEMAS = {"modebench-saved-responses-v3", "modebench-saved-responses-v4"}
MAX_RESULT_LINE_BYTES = 64 * 1024 * 1024


def read_report(path: str | Path, *, verify_results: bool = True) -> dict[str, Any]:
    path = Path(path)
    payload = loads(path.read_bytes())
    if not isinstance(payload, dict) or payload.get("schema") not in SCHEMAS:
        raise InputError("report requires a ModeBench v3 or v4 receipt")
    if not isinstance(payload.get("evaluation"), dict):
        raise InputError("report requires evaluation metadata")
    status = payload["evaluation"].get("status")
    if status not in ("completed", "failed"):
        raise InputError("report has no valid evaluation status")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise InputError("report cells must be a list")
    try:
        for cell in cells:
            normalize_task_identity(cell["level"], cell["domain"])
            k = positive_integer(cell["k"], "report k")
            notes = cell.get("protocol_notes", [])
            if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
                raise InputError("invalid protocol notes")
            for key in ("accuracy", "pass_at_k", "distinct_at_k"):
                value = cell[key]
                if value is not None and (type(value) not in (int, float) or value < 0):
                    raise InputError(f"invalid report metric: {key}")
                maximum = k if key == "distinct_at_k" else 1
                if value is not None and value > maximum:
                    raise InputError(f"report metric out of bounds: {key}")
                if status == "completed" and value is None:
                    raise InputError("completed report has missing aggregate scores")
                if status == "failed" and value is not None:
                    raise InputError("failed report must not contain aggregate scores")
            diversity = cell["pcmd"]["d_mode"]
            if diversity is not None and (
                type(diversity) not in (int, float) or not 0 <= diversity <= 1
            ):
                raise InputError("invalid PCMD")
            if status == "failed" and (diversity is not None or cell["pcmd"]["reportable"]):
                raise InputError("failed report must not contain reportable PCMD")
            if not isinstance(cell["pcmd"]["reportable"], bool):
                raise InputError("invalid reportability flag")
    except (TypeError, KeyError) as error:
        raise InputError(f"incomplete report cell: {error}") from error
    if payload["schema"] == "modebench-saved-responses-v4":
        reference = payload.get("prompt_results")
        if (
            not isinstance(reference, dict)
            or reference.get("format") != "modebench-prompt-result-v1"
        ):
            raise InputError("invalid detailed-result reference")
        if not isinstance(reference.get("path"), str) or not reference["path"]:
            raise InputError("missing detailed-result path")
        if not isinstance(reference.get("sha256"), str) or not re.fullmatch(
            "[0-9a-f]{64}", reference["sha256"]
        ):
            raise InputError("missing detailed-result hash")
        if type(reference.get("records")) is not int or reference["records"] < 1:
            raise InputError("invalid detailed-result count")
        if verify_results and file_sha256(path.parent / reference["path"]) != reference["sha256"]:
            raise InputError("detailed-result journal hash mismatch")
    return payload


def iter_prompt_results(path: str | Path) -> Iterator[dict[str, Any]]:
    """Verify the journal before yielding; preserve original prompt order in v4."""
    path = Path(path)
    payload = read_report(path)
    if payload["schema"] == "modebench-saved-responses-v3":
        for cell in payload["cells"]:
            yield from cell["prompt_results"]
        return
    reference = payload["prompt_results"]
    count = 0
    with (path.parent / reference["path"]).open("rb") as handle:
        while True:
            line = handle.readline(MAX_RESULT_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_RESULT_LINE_BYTES:
                raise InputError("detailed-result line exceeds byte limit")
            entry = loads(line)
            if (
                entry.get("schema") != "modebench-prompt-result-v1"
                or entry.get("sequence") != count
            ):
                raise InputError("detailed-result sequence mismatch")
            count += 1
            yield entry["result"]
    if count != reference["records"]:
        raise InputError("detailed-result count mismatch")
