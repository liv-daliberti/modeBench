import hashlib
import json
import random
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from modebench.evaluation import evaluate
from modebench.metrics import cell_summary
from modebench.streaming import CellAccumulator, atomic_output, evaluate_file
from modebench.validation import InputError

ROOT = Path(__file__).resolve().parents[1]
DEMO = json.loads((ROOT / "src/modebench/walkthrough.json").read_text())


def write_input(tmp_path, count=15):
    path = tmp_path / "responses.jsonl"
    records = [{**deepcopy(DEMO["records"][i % 5]), "id": str(i)} for i in range(count)]
    path.write_text("\n" + "\n\n".join(json.dumps(r) for r in records) + "\n")
    return path, records


def details(output):
    report = json.loads(output.read_text())
    journal = output.parent / report["prompt_results"]["path"]
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == report["prompt_results"]["sha256"]
    return report, [json.loads(line) for line in journal.read_text().splitlines()]


def assert_equivalent(expected, actual, entries):
    for key in ("records_sha256", "run", "run_sha256", "dataset", "evaluation"):
        assert actual[key] == expected[key]
    for left, right in zip(expected["cells"], actual["cells"]):
        for key in (
            "level",
            "domain",
            "k",
            "accuracy",
            "pass_at_k",
            "distinct_at_k",
            "pcmd",
            "protocol_notes",
        ):
            assert right[key] == left[key]
        matching = [
            e["result"]
            for e in entries
            if (e["level"], e["domain"], e["k"]) == (left["level"], left["domain"], left["k"])
        ]
        assert matching == left["prompt_results"]


def test_stream_matches_original_receipt_and_writes_incremental_details(tmp_path):
    source, records = write_input(tmp_path)
    output = tmp_path / "report.json"
    evaluate_file(source, output, run=DEMO["run"])
    actual, entries = details(output)
    assert actual["schema"] == "modebench-saved-responses-v4"
    assert_equivalent(evaluate(records, run=DEMO["run"]), actual, entries)


@pytest.mark.parametrize("when", ["grading", "finalizing"])
def test_resume_reuses_committed_results_and_repairs_journal(tmp_path, monkeypatch, when):
    source, records = write_input(tmp_path)
    output = tmp_path / "report.json"

    def interrupt(event):
        if event["phase"] == when and event["completed_prompts"] >= 4:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        evaluate_file(source, output, run=DEMO["run"], progress=interrupt)
    assert not output.exists()
    work = output.with_name(output.name + ".work")
    before = (
        sqlite3.connect(work / "ledger.sqlite3")
        .execute("SELECT count(*) FROM prompts WHERE result_json IS NOT NULL")
        .fetchone()[0]
    )
    journal = work / "results.jsonl"
    journal.write_text(journal.read_text() + "{partial uncommitted tail")
    from modebench import evaluation

    original = evaluation.grade_prepared
    calls = []

    def recorded(prepared):
        calls.append(prepared["id"])
        return original(prepared)

    monkeypatch.setattr(evaluation, "grade_prepared", recorded)
    evaluate_file(source, output, run=DEMO["run"], resume=True)
    assert len(calls) == len(records) - before
    actual, entries = details(output)
    assert_equivalent(evaluate(records, run=DEMO["run"]), actual, entries)


@pytest.mark.parametrize("change", ["input", "threshold", "run", "code", "reference"])
def test_resume_rejects_changed_identity(tmp_path, monkeypatch, change):
    source, records = write_input(tmp_path)
    output = tmp_path / "report.json"

    def interrupt(event):
        if event["phase"] == "grading" and event["completed_prompts"] == 2:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        evaluate_file(source, output, run=DEMO["run"], progress=interrupt)
    options = {"run": DEMO["run"], "resume": True}
    if change == "input":
        source.write_text(source.read_text() + "\n")
    if change == "reference":
        records[0]["answer"]["target"] += 1
        source.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    if change == "threshold":
        options["min_defined_prompts"] = 3
    if change == "run":
        options["run"] = deepcopy(DEMO["run"])
        options["run"]["model"]["revision"] = "other"
    if change == "code":
        from modebench import streaming

        original = streaming.software_identity
        monkeypatch.setattr(
            streaming,
            "software_identity",
            lambda: {**original(), "package_source_sha256": "changed"},
        )
    with pytest.raises(InputError, match="identity mismatch"):
        evaluate_file(source, output, **options)
    assert not output.exists()


def test_all_input_is_preflighted_before_grading(tmp_path, monkeypatch):
    source, records = write_input(tmp_path)
    records[-1]["unexpected"] = "bad"
    source.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def forbidden(*args):
        raise AssertionError("graded before preflight finished")

    monkeypatch.setattr("modebench.evaluation.grade_prepared", forbidden)
    with pytest.raises(InputError, match="unknown fields"):
        evaluate_file(source, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_late_duplicate_is_rejected_by_disk_seen(tmp_path):
    source, records = write_input(tmp_path)
    source.write_text(source.read_text() + json.dumps(records[0]) + "\n")
    with pytest.raises(InputError, match="duplicate prompt"):
        evaluate_file(source, tmp_path / "out.json")


def test_checkpoint_corruption_is_not_reused(tmp_path):
    source, _ = write_input(tmp_path)
    output = tmp_path / "out.json"

    def interrupt(event):
        if event["phase"] == "grading" and event["completed_prompts"] == 1:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        evaluate_file(source, output, progress=interrupt)
    with sqlite3.connect(str(output) + ".work/ledger.sqlite3") as connection:
        connection.execute("UPDATE prompts SET result_json='{}' WHERE seq=0")
    with pytest.raises(InputError, match="result identity"):
        evaluate_file(source, output, resume=True)


def test_atomic_publication_refuses_overwrite_and_cleans_interruption(tmp_path):
    output = tmp_path / "out.json"
    with pytest.raises(KeyboardInterrupt):
        with atomic_output(output) as handle:
            handle.write("partial")
            raise KeyboardInterrupt()
    assert not list(tmp_path.iterdir())
    output.write_text("original")
    with pytest.raises(FileExistsError):
        with atomic_output(output) as handle:
            handle.write("replacement")
    assert output.read_text() == "original"
    assert len(list(tmp_path.iterdir())) == 1


def test_constant_space_accumulator_matches_reference_metrics():
    randomizer = random.Random(43)
    for _ in range(40):
        prompts = []
        acc = CellAccumulator()
        for _ in range(randomizer.randrange(1, 80)):
            attempts = [
                {"verified": key is not None, "canonical_key": key}
                for key in (
                    randomizer.choice(["a", "b", "c", None])
                    for _ in range(randomizer.randrange(1, 12))
                )
            ]
            keys = [a["canonical_key"] for a in attempts if a["verified"]]
            p = {
                "accuracy": len(keys) / len(attempts),
                "pass_at_k": float(bool(keys)),
                "distinct_at_k": len(set(keys)),
                "draws": [{"attempts": attempts}],
            }
            prompts.append(p)
            acc.add(p)
        actual = acc.summary(3, True)["pcmd"]
        expected = cell_summary(prompts, min_defined_prompts=3)
        for key in expected:
            if isinstance(expected[key], float):
                assert actual[key] == pytest.approx(expected[key], rel=1e-14, abs=1e-15)
            else:
                assert actual[key] == expected[key]


def test_parallel_and_resumed_parallel_are_identical_to_serial(tmp_path):
    source, records = write_input(tmp_path, 30)
    serial = tmp_path / "serial.json"
    parallel = tmp_path / "parallel.json"
    resumed = tmp_path / "resumed.json"
    evaluate_file(source, serial, run=DEMO["run"])
    evaluate_file(source, parallel, run=DEMO["run"], workers=2)

    def interrupt(event):
        if event["phase"] == "grading" and event["completed_prompts"] == 7:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        evaluate_file(source, resumed, run=DEMO["run"], workers=2, progress=interrupt)
    # Concurrency is operational, and may change on resume without changing the estimand.
    evaluate_file(source, resumed, run=DEMO["run"], workers=1, resume=True)
    expected = evaluate(records, run=DEMO["run"])
    for output in (serial, parallel, resumed):
        report, entries = details(output)
        assert_equivalent(expected, report, entries)
    assert details(serial)[1] == details(parallel)[1] == details(resumed)[1]


def test_parallel_workers_are_reaped_on_interruption(tmp_path, monkeypatch):
    from modebench import parallel

    processes = []
    original = parallel.BoundedWorker

    def worker():
        instance = original()
        processes.append(instance)
        return instance

    monkeypatch.setattr(parallel, "BoundedWorker", worker)
    source, _ = write_input(tmp_path, 20)

    def interrupt(event):
        if event["phase"] == "grading" and event["completed_prompts"] == 2:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        evaluate_file(source, tmp_path / "out.json", workers=2, progress=interrupt)
    assert processes and all(p._process is None for p in processes)


def test_journal_mutation_prevents_final_publication(tmp_path):
    source, _ = write_input(tmp_path)
    output = tmp_path / "out.json"

    def corrupt(event):
        if event["phase"] == "finalizing":
            (tmp_path / "out.json.work/results.jsonl").write_text("corrupt")

    with pytest.raises(InputError, match="journal changed"):
        evaluate_file(source, output, progress=corrupt)
    assert not output.exists()
    evaluate_file(source, output, resume=True)
    details(output)


def test_full_frozen_split_matches_in_memory_reference(tmp_path):
    from tests.test_frozen_evaluation import CONFIG, DATA, RUN, response

    records = [response(i) for i in range(128)]
    source = tmp_path / "frozen.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    output = tmp_path / "out.json"
    options = dict(config=CONFIG, data_root=DATA, run=RUN)
    evaluate_file(source, output, workers=2, **options)
    report, entries = details(output)
    assert_equivalent(evaluate(records, **options), report, entries)


def test_checkpoint_lock_rejects_concurrent_writer(tmp_path):
    from modebench.streaming import checkpoint

    work = tmp_path / "work"
    with checkpoint(work, False):
        with pytest.raises(InputError, match="in use"):
            with checkpoint(work, True):
                pass


def test_preflight_interruption_restarts_validation_without_grading(tmp_path):
    source, records = write_input(tmp_path, 110)
    output = tmp_path / "out.json"

    def interrupt(event):
        if event["phase"] == "validating" and event["completed_prompts"] == 100:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        evaluate_file(source, output, run=DEMO["run"], progress=interrupt)
    assert not output.exists()
    evaluate_file(source, output, run=DEMO["run"], resume=True)
    report, entries = details(output)
    assert_equivalent(evaluate(records, run=DEMO["run"]), report, entries)


def test_resume_preserves_committed_evaluator_failures(tmp_path, monkeypatch):
    from modebench import evaluation
    from modebench.diagnostics import result

    source, _ = write_input(tmp_path, 10)
    output = tmp_path / "out.json"
    original = evaluation.grade_response
    monkeypatch.setattr(
        evaluation, "grade_response", lambda *_: result("timeout", detail="injected")
    )

    def interrupt(event):
        if event["phase"] == "grading" and event["completed_prompts"] == 1:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        evaluate_file(source, output, progress=interrupt)
    monkeypatch.setattr(evaluation, "grade_response", original)
    evaluate_file(source, output, resume=True)
    report, entries = details(output)
    assert report["evaluation"]["status"] == "failed"
    assert report["evaluation"]["status_counts"]["timeout"] == 4
    assert all(cell["accuracy"] is None for cell in report["cells"])
    assert all(a["status"] == "timeout" for a in entries[0]["result"]["draws"][0]["attempts"])


def test_sigkill_recovery_uses_durable_prefix(tmp_path):
    import subprocess
    import sys

    source, records = write_input(tmp_path, 100)
    output = tmp_path / "out.json"
    command = [
        sys.executable,
        "-m",
        "modebench.cli",
        "evaluate",
        str(source),
        "--output",
        str(output),
        "--progress-every",
        "1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for line in process.stderr:
            event = json.loads(line)
            if event["phase"] == "grading" and event["completed_prompts"] >= 3:
                process.kill()
                break
        process.wait(timeout=10)
        assert process.returncode < 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()
    assert not output.exists()
    evaluate_file(source, output, resume=True)
    report, entries = details(output)
    assert_equivalent(evaluate(records), report, entries)


@pytest.mark.parametrize("change", ["cell", "metadata"])
def test_checkpoint_grouping_and_metadata_corruption_are_detected(tmp_path, change):
    source, _ = write_input(tmp_path)
    output = tmp_path / "out.json"

    def interrupt(event):
        if event["phase"] == "grading" and event["completed_prompts"] == 1:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        evaluate_file(source, output, progress=interrupt)
    with sqlite3.connect(str(output) + ".work/ledger.sqlite3") as connection:
        if change == "cell":
            connection.execute("UPDATE prompts SET k=99 WHERE seq=0")
        else:
            connection.execute("UPDATE info SET value='{}' WHERE key='dataset'")
    with pytest.raises(InputError, match="identity mismatch"):
        evaluate_file(source, output, resume=True)
    assert not output.exists()
