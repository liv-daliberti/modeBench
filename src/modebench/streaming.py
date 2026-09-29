"""Bounded-memory JSONL evaluation with a durable, identity-bound SQLite ledger."""

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from . import evaluation
from .api_types import Progress
from .diagnostics import UNSCORABLE
from .identity import digest_json, software_identity
from .metrics import POOLED, effective_modes, prompt_mode_diversity
from .validation import InputError, loads, positive_integer

MAX_RESPONSES_PER_PROMPT = 4096
MAX_INPUT_LINE_BYTES = 16 * 1024 * 1024
CHECKPOINT_SCHEMA = "modebench-checkpoint-v1"
REPORT_SCHEMA = "modebench-saved-responses-v4"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path):
    """Yield bounded raw lines, including blanks, so input identities cover every byte."""
    with Path(path).open("rb") as handle:
        number = 0
        while True:
            line = handle.readline(MAX_INPUT_LINE_BYTES + 1)
            if not line:
                break
            number += 1
            if len(line) > MAX_INPUT_LINE_BYTES:
                raise InputError(f"line {number}: exceeds {MAX_INPUT_LINE_BYTES} byte input limit")
            yield number, line


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_output(path, *, replace=False):
    """Publish complete, fsynced bytes; ordinary outputs never overwrite an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + "-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DiskSeen:
    def __init__(self, connection):
        self.connection = connection

    def __contains__(self, key):
        return (
            self.connection.execute(
                "SELECT 1 FROM seen WHERE level=? AND domain=? AND rid=?", key
            ).fetchone()
            is not None
        )

    def add(self, key):
        self.connection.execute("INSERT INTO seen VALUES (?,?,?)", key)


def _set_info(connection, key, value):
    connection.execute("INSERT OR REPLACE INTO info VALUES (?,?)", (key, canonical(value)))


def _info(connection, key):
    row = connection.execute("SELECT value FROM info WHERE key=?", (key,)).fetchone()
    if row is None:
        raise InputError(f"incomplete checkpoint metadata: {key}")
    return loads(row[0])


@contextmanager
def checkpoint(directory, resume):
    directory = Path(directory)
    if resume:
        if not (directory / "ledger.sqlite3").is_file():
            raise InputError(f"no checkpoint to resume: {directory}")
    else:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise InputError(
                f"checkpoint already exists: {directory}; use --resume or a new destination"
            ) from error
    with (directory / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise InputError("checkpoint is in use by another evaluator") from error
        connection = sqlite3.connect(directory / "ledger.sqlite3")
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA cache_size=-4096")
            connection.execute("PRAGMA temp_store=FILE")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS seen (level INTEGER, domain TEXT, rid TEXT,
                                                PRIMARY KEY(level,domain,rid));
                CREATE TABLE IF NOT EXISTS prompts (
                    seq INTEGER PRIMARY KEY, level INTEGER, domain TEXT, k INTEGER,
                    raw_json TEXT NOT NULL, prepared_json TEXT NOT NULL, prepared_sha TEXT,
                    result_json TEXT, result_sha TEXT);
                CREATE INDEX IF NOT EXISTS cells ON prompts(level,domain,k,seq);
            """)
            yield connection
        except sqlite3.DatabaseError as error:
            raise InputError(f"checkpoint database error: {error}") from error
        finally:
            connection.close()
            fcntl.flock(lock, fcntl.LOCK_UN)


def _notify(callback, phase, completed, total=None):
    if callback is not None:
        callback({"phase": phase, "completed_prompts": completed, "total_prompts": total})


def _preflight(connection, context, path, identity, progress):
    # Interrupted validation can restart safely: no candidates have been graded yet.
    connection.execute("DELETE FROM prompts")
    connection.execute("DELETE FROM seen")
    records_digest, input_digest = hashlib.sha256(), hashlib.sha256()
    records_digest.update(b"[")
    count = 0
    for number, line in iter_jsonl(path):
        input_digest.update(line)
        if not line.strip():
            continue
        try:
            raw = loads(line)
        except ValueError as error:
            raise InputError(f"line {number}: {error}") from error
        if (
            isinstance(raw, dict)
            and isinstance(raw.get("responses"), list)
            and len(raw["responses"]) > MAX_RESPONSES_PER_PROMPT
        ):
            raise InputError(
                f"line {number}: exceeds {MAX_RESPONSES_PER_PROMPT} responses per prompt"
            )
        prepared = context.accept(raw, f"line {number}")
        encoded = canonical(raw)
        if count:
            records_digest.update(b",")
        records_digest.update(encoded.encode())
        connection.execute(
            "INSERT INTO prompts(seq,level,domain,k,raw_json,prepared_json) VALUES (?,?,?,?,?,?)",
            (
                count,
                prepared["level"],
                prepared["domain"],
                len(raw["responses"]),
                encoded,
                canonical(prepared),
            ),
        )
        count += 1
        if count % 100 == 0:
            connection.commit()
            _notify(progress, "validating", count)
    records_digest.update(b"]")
    if input_digest.hexdigest() != identity["input_sha256"]:
        raise InputError("input changed while validation was running; use a fresh checkpoint")
    context.finish()
    for seq, text in connection.execute("SELECT seq,prepared_json FROM prompts ORDER BY seq"):
        prepared = context.bind(loads(text))
        encoded = canonical(prepared)
        connection.execute(
            "UPDATE prompts SET prepared_json=?,prepared_sha=? WHERE seq=?",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest(), seq),
        )
    _set_info(connection, "records_sha256", records_digest.hexdigest())
    _set_info(connection, "run", context.run)
    _set_info(connection, "dataset", context.dataset_identity())
    _set_info(connection, "count", count)
    _set_info(connection, "preflight_sha256", _preflight_digest(connection))
    _set_info(connection, "phase", "grading")
    connection.commit()
    _notify(progress, "validated", count, count)


def _preflight_digest(connection):
    return digest_json(
        {key: _info(connection, key) for key in ("records_sha256", "run", "dataset", "count")}
    )


def _check_ledger(connection):
    """Detect missing/reordered entries and corruption before accepting prior results."""
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise InputError("checkpoint integrity check failed")
    if _preflight_digest(connection) != _info(connection, "preflight_sha256"):
        raise InputError("checkpoint preflight metadata identity mismatch")
    records_digest = hashlib.sha256(b"[")
    completed, saw_pending, count = 0, False, 0
    for (
        seq,
        raw,
        prepared,
        prepared_sha,
        result,
        result_sha,
        level,
        domain,
        k,
    ) in connection.execute(
        "SELECT seq,raw_json,prepared_json,prepared_sha,result_json,result_sha,level,domain,k "
        "FROM prompts ORDER BY seq"
    ):
        if seq != count or hashlib.sha256(prepared.encode()).hexdigest() != prepared_sha:
            raise InputError("checkpoint prepared-record identity mismatch")
        record = loads(prepared)
        if (level, domain, k) != (
            record["level"],
            record["domain"],
            len(record["raw"]["responses"]),
        ):
            raise InputError("checkpoint cell identity mismatch")
        if canonical(record["raw"]) != raw:
            raise InputError("checkpoint input snapshot mismatch")
        if count:
            records_digest.update(b",")
        records_digest.update(raw.encode())
        count += 1
        if result is None:
            saw_pending = True
            if result_sha is not None:
                raise InputError("checkpoint incomplete result identity")
        else:
            if saw_pending or hashlib.sha256(result.encode()).hexdigest() != result_sha:
                raise InputError("checkpoint result identity or ordering mismatch")
            completed += 1
    records_digest.update(b"]")
    if count != _info(connection, "count") or records_digest.hexdigest() != _info(
        connection, "records_sha256"
    ):
        raise InputError("checkpoint input record digest mismatch")
    return completed


def _entry(seq, prepared, result):
    return {
        "schema": "modebench-prompt-result-v1",
        "sequence": seq,
        "level": prepared["level"],
        "domain": prepared["domain"],
        "k": len(prepared["raw"]["responses"]),
        "record_sha256": digest_json(prepared["raw"]),
        "result": result,
    }


def _repair_journal(connection, path):
    # The transaction ledger is authoritative. Rebuild a truncated, extra, or missing tail.
    digest = hashlib.sha256()
    with atomic_output(path, replace=True) as handle:
        for seq, prepared, result in connection.execute(
            "SELECT seq,prepared_json,result_json FROM prompts WHERE result_json IS NOT NULL ORDER BY seq"
        ):
            line = canonical(_entry(seq, loads(prepared), loads(result))) + "\n"
            handle.write(line)
            digest.update(line.encode())
    return digest


class CellAccumulator:
    """Exact sums of binary floats, with constant-space variance accumulation."""

    def __init__(self):
        self.prompts = self.defined = 0
        self.accuracy = self.passes = self.distinct = Fraction(0)
        self.diversity = self.squares = Fraction(0)

    def add(self, prompt):
        self.prompts += 1
        if prompt["accuracy"] is None:
            return
        self.accuracy += Fraction(prompt["accuracy"])
        self.passes += Fraction(prompt["pass_at_k"])
        self.distinct += Fraction(prompt["distinct_at_k"])
        value = prompt_mode_diversity(prompt["draws"], POOLED)
        if value is not None:
            value = Fraction(value)
            self.defined += 1
            self.diversity += value
            self.squares += value * value

    def summary(self, threshold, healthy):
        if not healthy:
            return {
                "accuracy": None,
                "pass_at_k": None,
                "distinct_at_k": None,
                "pcmd": {"d_mode": None, "reportable": False, "reason": "verifier_failure"},
            }
        n, defined = self.prompts, self.defined
        mean = float(self.diversity) / defined if defined else None
        error = None
        if defined > 1:
            variance = (self.squares - self.diversity**2 / defined) / (defined - 1)
            error = math.sqrt(float(variance)) / defined**0.5
        return {
            "accuracy": float(self.accuracy) / n,
            "pass_at_k": float(self.passes) / n,
            "distinct_at_k": float(self.distinct) / n,
            "pcmd": {
                "aggregation": POOLED,
                "d_mode": mean,
                "defined_prompts": defined,
                "prompts": n,
                "support": defined / n,
                "min_defined_prompts": threshold,
                "reportable": defined >= threshold,
                "standard_error": error,
                "rarefied_distinct_at_2": 1 + mean if mean is not None else None,
                "effective_modes": effective_modes(mean),
            },
        }


def _write_receipt(connection, output, directory, software, threshold, identity):
    status_counts = Counter()
    for (text,) in connection.execute("SELECT result_json FROM prompts ORDER BY seq"):
        result = loads(text)
        status_counts.update(a["status"] for a in result["draws"][0]["attempts"])
    healthy = not any(status_counts[s] for s in UNSCORABLE)
    header = evaluation.receipt_header(
        records_hash=_info(connection, "records_sha256"),
        run=_info(connection, "run"),
        dataset=_info(connection, "dataset"),
        threshold=threshold,
        status_counts=status_counts,
        created_at=_info(connection, "created_at"),
        software=software,
    )
    header["schema"] = REPORT_SCHEMA
    header["input_sha256"] = identity["input_sha256"]
    header["execution"] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "identity_sha256": digest_json(identity),
        "history": _info(connection, "history"),
    }
    header["prompt_results"] = {
        "format": "modebench-prompt-result-v1",
        "records": _info(connection, "count"),
        "path": os.path.relpath(directory / "results.jsonl", output.parent),
        "sha256": file_sha256(directory / "results.jsonl"),
    }
    cell_count = 0
    with atomic_output(output) as handle:
        handle.write(canonical(header)[:-1] + ',"cells":[')
        for level, domain, k in connection.execute(
            "SELECT DISTINCT level,domain,k FROM prompts ORDER BY level,domain,k"
        ):
            accumulator = CellAccumulator()
            for (text,) in connection.execute(
                "SELECT result_json FROM prompts WHERE level=? AND domain=? AND k=? ORDER BY seq",
                (level, domain, k),
            ):
                accumulator.add(loads(text))
            cell = {
                "level": level,
                "domain": domain,
                "k": k,
                **accumulator.summary(threshold, healthy),
                "protocol_notes": evaluation.protocol_notes(level, domain),
            }
            if cell_count:
                handle.write(",")
            handle.write(canonical(cell))
            cell_count += 1
        handle.write("]}\n")
    return {
        "output": str(output),
        "cells": cell_count,
        "prompts": _info(connection, "count"),
        "dataset_kind": header["dataset"]["kind"],
        "status": header["evaluation"]["status"],
        "work_dir": str(directory),
    }


def evaluate_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_defined_prompts: int = 30,
    run: Any = None,
    data_root: str | Path | None = None,
    config: str | None = None,
    split: str = "eval",
    allow_partial: bool = False,
    work_dir: str | Path | None = None,
    resume: bool = False,
    progress: Callable[[Progress], None] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Validate and grade a JSONL snapshot; commit each prompt once and publish atomically.

    The final v4 receipt is small; detailed prompt results live in a hash-bound JSONL
    journal beside the SQLite checkpoint. Keep the journal with the receipt.
    """
    workers = positive_integer(workers, "workers")
    from .parallel import MAX_WORKERS, ParallelGrader

    if workers > MAX_WORKERS:
        raise InputError(f"workers must be between 1 and {MAX_WORKERS}")
    input_path, output = Path(input_path).resolve(), Path(output_path).resolve()
    directory = (
        Path(work_dir).resolve()
        if work_dir is not None
        else output.with_name(output.name + ".work")
    )
    if output.exists():
        raise InputError(f"output already exists: {output}")
    if directory == output or directory == input_path or directory in input_path.parents:
        raise InputError("checkpoint must not contain the input file or occupy an output path")
    software = software_identity()
    context_options = dict(
        min_defined_prompts=min_defined_prompts,
        run=run,
        data_root=data_root,
        config=config,
        split=split,
        allow_partial=allow_partial,
    )
    with checkpoint(directory, resume) as connection:
        context = evaluation.Preflight(**context_options, seen=DiskSeen(connection))
        identity = {
            "schema": CHECKPOINT_SCHEMA,
            "input_sha256": file_sha256(input_path),
            "options": {k: v for k, v in context_options.items() if k != "data_root"},
            "frozen_split": context.metadata,
            "manifest_sha256": file_sha256(context.root / "manifest.json") if config else None,
            "software": {k: v for k, v in software.items() if k != "git"},
            "input_line_limit": MAX_INPUT_LINE_BYTES,
            "responses_per_prompt_limit": MAX_RESPONSES_PER_PROMPT,
        }
        if resume:
            if _info(connection, "identity") != identity:
                raise InputError(
                    "resume identity mismatch: input, configuration, dataset, or evaluator changed"
                )
            if _info(connection, "output") != str(output):
                raise InputError("resume output destination differs from checkpoint")
        else:
            _set_info(connection, "identity", identity)
            _set_info(connection, "output", str(output))
            _set_info(connection, "phase", "preflight")
            _set_info(connection, "created_at", datetime.now(timezone.utc).isoformat())
            _set_info(connection, "history", [])
            connection.commit()
        if _info(connection, "phase") == "preflight":
            _notify(progress, "validating", 0)
            _preflight(connection, context, input_path, identity, progress)
        completed = _check_ledger(connection)
        count = _info(connection, "count")
        journal_digest = _repair_journal(connection, directory / "results.jsonl")
        history = _info(connection, "history")
        history.append(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "workers": workers,
                "resumed_prompts": completed,
            }
        )
        _set_info(connection, "history", history)
        connection.commit()
        _notify(progress, "grading", completed, count)
        prompts = (
            (seq, loads(text))
            for seq, text in connection.execute(
                "SELECT seq,prepared_json FROM prompts WHERE result_json IS NULL ORDER BY seq"
            )
        )
        with ParallelGrader(workers) if workers > 1 else nullcontext() as pool:
            outcomes = (
                pool.results(prompts)
                if pool is not None
                else (
                    (seq, prepared, evaluation.grade_prepared(prepared))
                    for seq, prepared in prompts
                )
            )
            with (directory / "results.jsonl").open("a", encoding="utf-8") as journal:
                for seq, prepared, outcome in outcomes:
                    encoded = canonical(outcome)
                    connection.execute(
                        "UPDATE prompts SET result_json=?,result_sha=? WHERE seq=?",
                        (encoded, hashlib.sha256(encoded.encode()).hexdigest(), seq),
                    )
                    connection.commit()
                    line = canonical(_entry(seq, prepared, outcome)) + "\n"
                    journal.write(line)
                    journal.flush()
                    journal_digest.update(line.encode())
                    completed += 1
                    _notify(progress, "grading", completed, count)
                os.fsync(journal.fileno())
        _notify(progress, "finalizing", count, count)
        if file_sha256(input_path) != identity["input_sha256"]:
            raise InputError("input changed during evaluation; no final output published")
        if file_sha256(directory / "results.jsonl") != journal_digest.hexdigest():
            raise InputError("result journal changed during evaluation; no final output published")
        summary = _write_receipt(
            connection, output, directory, software, context.threshold, identity
        )
        _set_info(connection, "phase", "complete")
        _set_info(connection, "output_sha256", file_sha256(output))
        connection.commit()
        _notify(progress, "complete", count, count)
        return summary
