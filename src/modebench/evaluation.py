"""Shared evaluation contracts for the in-memory and checkpointed file APIs."""

import hashlib
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .contracts import normalize_task_identity
from .data import load_split, prompt_id, split_record
from .diagnostics import UNSCORABLE, VerifierExecutionError
from .identity import digest_json, software_identity
from .metrics import POOLED, cell_summary
from .prompts import HISTORICAL, make_messages, profile_metadata, prompt_sha256
from .validation import InputError, json_value, positive_integer, validate_reference, validate_run
from .verifier import grade_response

RECORD_FIELDS = {
    "id",
    "level",
    "domain",
    "answer",
    "responses",
    "problem",
    "run",
    "messages",
    "prompt_sha256",
    "dataset_sha256",
}


def protocol_notes(level, domain):
    notes = []
    if level == 4:
        notes.append("Level 4 as a whole is not admitted.")
        if domain == "mathir":
            notes.append("Level 4 MathIR is not difficulty-matched.")
    return notes


class Preflight:
    """Shared input contracts; seen may be a disk-backed unique-key collection."""

    def __init__(
        self,
        *,
        min_defined_prompts=30,
        run=None,
        data_root=None,
        config=None,
        split="eval",
        allow_partial=False,
        seen=None,
    ):
        self.threshold = positive_integer(min_defined_prompts, "min_defined_prompts")
        if type(allow_partial) is not bool:
            raise InputError("allow_partial must be a boolean")
        if config is None and (data_root is not None or allow_partial or split != "eval"):
            raise InputError("data_root, split, and allow_partial require a frozen config")
        self.shared_run = run
        self.run = validate_run(run) if run is not None else None
        self.seen = set() if seen is None else seen
        self.inline_present = self.inline_missing = False
        self.count = 0
        self.frozen_k = None
        self.config, self.split, self.allow_partial = config, split, allow_partial
        self.metadata, self.dataset = None, None
        self.expected_ids, self.missing = [], None
        self.root = Path(data_root or "data")
        if config is not None:
            self.metadata = split_record(self.root, config, split, frozen=True)
            self.dataset = load_split(self.root, config, split, frozen=True)
            self.expected_ids = [prompt_id(config, split, i) for i in range(len(self.dataset))]
            self.row_indices = {rid: i for i, rid in enumerate(self.expected_ids)}

    def accept(self, raw, location):
        label = (
            f"{location}, prompt {str(raw.get('id', '?'))[:100]!r}"
            if isinstance(raw, dict)
            else location
        )
        try:
            if not isinstance(raw, dict):
                raise InputError("each record must be an object")
            json_value(raw, "record")
            unknown = raw.keys() - RECORD_FIELDS
            if unknown:
                raise InputError(
                    f"unknown fields: {', '.join(sorted(unknown))}; put generation metadata in run"
                )
            for field in ("id", "responses"):
                if field not in raw:
                    raise InputError(f"missing field: {field}")
            rid = raw["id"]
            if isinstance(rid, bool) or not isinstance(rid, (str, int)) or not str(rid).strip():
                raise InputError("id must be a nonempty string or integer")
            rid = str(rid)
            responses = raw["responses"]
            if (
                not isinstance(responses, list)
                or not responses
                or not all(isinstance(s, str) for s in responses)
            ):
                raise InputError(
                    "responses must be a nonempty list of strings (retain failed draws)"
                )
            if self.metadata is not None:
                if rid not in self.row_indices:
                    raise InputError(f"unknown frozen prompt ID: {rid}")
                if "answer" in raw:
                    raise InputError(
                        "frozen evaluation resolves answer from the dataset; omit supplied answer"
                    )
                level, domain = self.metadata["level"], self.metadata["domain"]
                if normalize_task_identity(raw.get("level", level), raw.get("domain", domain)) != (
                    level,
                    domain,
                ):
                    raise InputError("level/domain does not match the frozen configuration")
                row = dict(self.dataset[self.row_indices[rid]])
                if "problem" in raw and raw["problem"] != row["problem"]:
                    raise InputError("problem does not match the frozen row")
                if (
                    "dataset_sha256" in raw
                    and raw["dataset_sha256"] != self.metadata["parquet_sha256"]
                ):
                    raise InputError("dataset_sha256 does not match the frozen split")
                if self.frozen_k is not None and len(responses) != self.frozen_k:
                    raise InputError(
                        "frozen evaluation requires the same number of draws per prompt; retain failed draws"
                    )
                self.frozen_k = len(responses)
            else:
                for field in ("level", "domain", "answer"):
                    if field not in raw:
                        raise InputError(f"missing field: {field}")
                level, domain = normalize_task_identity(raw["level"], raw["domain"])
                row = dict(raw)
                if "dataset_sha256" in raw:
                    raise InputError("dataset_sha256 requires frozen evaluation with config")
            row["answer"] = validate_reference(level, domain, row["answer"])
            if "problem" in row and (
                not isinstance(row["problem"], str) or not row["problem"].strip()
            ):
                raise InputError("problem must be a nonempty string")
            key = level, domain, rid
            if key in self.seen:
                raise InputError(f"duplicate prompt: {key}")
            self.seen.add(key)
            inline = validate_run(raw["run"]) if "run" in raw else None
            self.inline_present |= inline is not None
            self.inline_missing |= inline is None
            if inline is not None:
                if self.run is None:
                    self.run = inline
                elif digest_json(inline) != digest_json(self.run):
                    raise InputError("mixed model or generation conditions: run metadata differ")
            self.count += 1
            # Discard irrelevant row metadata after reference and prompt binding.
            reference_row = {"answer": row["answer"]}
            if "problem" in row:
                reference_row["problem"] = row["problem"]
            return {
                "label": label,
                "raw": raw,
                "id": rid,
                "level": level,
                "domain": domain,
                "row": reference_row,
            }
        except VerifierExecutionError as error:
            raise VerifierExecutionError(
                {**error.diagnostic, "detail": f"{label}: {error}"}
            ) from error
        except (ValueError, TypeError, KeyError) as error:
            raise InputError(f"{label}: {error}") from error

    def finish(self):
        if not self.count:
            raise InputError("no prompt records")
        if self.shared_run is None and self.inline_present and self.inline_missing:
            raise InputError(
                "mixed declared/undeclared conditions: supply run metadata for the entire evaluation"
            )
        if self.metadata is not None:
            if self.run is None:
                raise InputError(
                    "frozen evaluation requires run metadata (model, generation, prompt_condition)"
                )
            if self.run["prompt_condition"] != HISTORICAL:
                raise InputError(
                    "frozen evaluation requires registered_hints_v1; use custom mode for changed conditions"
                )
            self.missing = [
                rid
                for rid in self.expected_ids
                if (self.metadata["level"], self.metadata["domain"], rid) not in self.seen
            ]
            if self.missing and not self.allow_partial:
                raise InputError(
                    f"incomplete frozen split: missing {len(self.missing)} of {len(self.expected_ids)} prompts; explicitly allow_partial to evaluate a subset"
                )

    def bind(self, prepared):
        raw, row = prepared["raw"], prepared["row"]
        level, domain, label = prepared["level"], prepared["domain"], prepared["label"]
        try:
            if self.run is not None and "draws_per_prompt" in self.run["generation"]:
                if len(raw["responses"]) != self.run["generation"]["draws_per_prompt"]:
                    raise InputError(
                        "responses count does not match run.generation.draws_per_prompt"
                    )
            condition = self.run["prompt_condition"] if self.run else None
            profile, prompt_hash = None, None
            if condition is not None:
                profile = profile_metadata(level, domain, condition)
                if "problem" in row:
                    prompt_hash = prompt_sha256(level, domain, row, condition)
            for field in ("prompt_sha256", "messages"):
                if field in raw:
                    if prompt_hash is None:
                        raise InputError(
                            f"{field} requires problem text and declared run.prompt_condition"
                        )
                    expected = (
                        prompt_hash
                        if field == "prompt_sha256"
                        else make_messages(level, domain, row, condition)
                    )
                    if raw[field] != expected:
                        raise InputError(f"{field} does not match the registered prompt")
            prepared["binding"] = {
                "reference_sha256": digest_json(row["answer"]),
                "prompt_sha256": prompt_hash,
                "prompt_profile": profile,
            }
            return prepared
        except (ValueError, TypeError, KeyError) as error:
            raise InputError(f"{label}: {error}") from error

    def dataset_identity(self):
        if self.metadata is None:
            return {"kind": "custom", "references_authenticated": False}
        missing = set(self.missing)
        identity = {
            "kind": "frozen",
            "references_authenticated": True,
            "registry": "modebench-portable-data-v1",
            "split_record": self.metadata,
            "manifest_sha256": hashlib.sha256(
                (self.root / "manifest.json").read_bytes()
            ).hexdigest(),
            "selected_prompt_ids": [rid for rid in self.expected_ids if rid not in missing],
            "missing_prompt_ids": self.missing,
            "complete": not self.missing,
            "selected_prompts": self.count,
            "expected_prompts": len(self.expected_ids),
            "allow_partial": self.allow_partial,
            "draws_per_prompt": self.frozen_k,
        }
        if self.config.endswith("_unique_answer"):
            identity["protocol_notes"] = [
                "Single-answer diagnostic; separate from the main benchmark."
            ]
        return identity


def grade_prepared(prepared, grader=None):
    """Grade one validated prompt using the registered scoring policy."""
    grader = grade_response if grader is None else grader
    attempts = [
        grader(prepared["level"], prepared["domain"], prepared["row"], text)
        for text in prepared["raw"]["responses"]
    ]
    failed = any(a["status"] in UNSCORABLE for a in attempts)
    keys = [a["canonical_key"] for a in attempts if a["verified"]]
    return {
        "id": prepared["id"],
        **prepared["binding"],
        "draws": [{"attempts": attempts}],
        "pass_at_k": None if failed else float(bool(keys)),
        "distinct_at_k": None if failed else len(set(keys)),
        "accuracy": None if failed else len(keys) / len(attempts),
    }


def receipt_header(
    *, records_hash, run, dataset, threshold, status_counts, created_at=None, software=None
):
    healthy = not any(status_counts.get(s, 0) for s in UNSCORABLE)
    return {
        "schema": "modebench-saved-responses-v3",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "records_sha256": records_hash,
        "software": software or software_identity(),
        "run": run,
        "generation_metadata_status": "user_declared" if run else "unreported",
        "run_sha256": digest_json(run) if run else None,
        "dataset": dataset,
        "evaluation": {
            "status": "completed" if healthy else "failed",
            "status_counts": dict(sorted(status_counts.items())),
            "scoring_policy": "wrong-or-malformed-count-as-failure; verifier-errors-suppress-all-aggregates-v1",
            "aggregation": POOLED,
            "groups_per_prompt": 1,
            "prompt_weighting": "equal",
            "min_defined_prompts": threshold,
            "pass_at_k": "empirical_saved_group",
            "prompt_hash_semantics": "expected_registered_messages; does not attest generation",
        },
    }


def evaluate(
    records,
    *,
    min_defined_prompts=30,
    run=None,
    data_root=None,
    config=None,
    split="eval",
    allow_partial=False,
    locations=None,
):
    """Evaluate an in-memory iterable. For large files use evaluate_file instead."""
    context = Preflight(
        min_defined_prompts=min_defined_prompts,
        run=run,
        data_root=data_root,
        config=config,
        split=split,
        allow_partial=allow_partial,
    )
    records = list(records)
    if locations is None:
        locations = [f"record {i}" for i in range(1, len(records) + 1)]
    if len(locations) != len(records):
        raise InputError("locations must match the number of records")
    prepared = [context.accept(raw, location) for raw, location in zip(records, locations)]
    context.finish()
    prepared = [context.bind(p) for p in prepared]
    cells, status_counts = defaultdict(list), Counter()
    for prompt in prepared:
        outcome = grade_prepared(prompt)
        attempts = outcome["draws"][0]["attempts"]
        status_counts.update(a["status"] for a in attempts)
        cells[(prompt["level"], prompt["domain"], len(attempts))].append(outcome)
    healthy = not any(status_counts[s] for s in UNSCORABLE)
    result = []
    for (level, domain, k), prompts in sorted(cells.items()):
        result.append(
            {
                "level": level,
                "domain": domain,
                "k": k,
                "pass_at_k": statistics.fmean(p["pass_at_k"] for p in prompts) if healthy else None,
                "distinct_at_k": statistics.fmean(p["distinct_at_k"] for p in prompts)
                if healthy
                else None,
                "accuracy": statistics.fmean(p["accuracy"] for p in prompts) if healthy else None,
                "pcmd": (
                    cell_summary(prompts, POOLED, min_defined_prompts=context.threshold)
                    if healthy
                    else {"d_mode": None, "reportable": False, "reason": "verifier_failure"}
                ),
                "protocol_notes": protocol_notes(level, domain),
                "prompt_results": prompts,
            }
        )
    return {
        **receipt_header(
            records_hash=digest_json(records),
            run=context.run,
            dataset=context.dataset_identity(),
            threshold=context.threshold,
            status_counts=status_counts,
        ),
        "cells": result,
    }
