import json
from importlib.resources import files
from pathlib import Path

import pytest

from modebench.api import (
    EvaluationOptions,
    Task,
    evaluate_file,
    grade,
    iter_results,
    load_tasks,
    make_prompt,
    read_report,
)
from modebench.validation import InputError

ROOT = Path(__file__).resolve().parents[1]


def test_complete_typed_public_workflow(tmp_path):
    tasks = [
        next(load_tasks("level1_" + domain, data_root=ROOT / "data"))
        for domain in ("countdown", "graph_coloring", "python_factors", "mathir", "pantry_plan")
    ]
    demo = json.loads(files("modebench").joinpath("walkthrough.json").read_text())
    records = {record["domain"]: record for record in demo["records"]}
    for task in tasks:
        assert isinstance(task, Task)
        messages = make_prompt(task)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert grade(task, records[task.domain]["responses"][0])["status"] == "correct"
    source = tmp_path / "responses.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in demo["records"]) + "\n")
    receipt = evaluate_file(
        source, tmp_path / "result.json", options=EvaluationOptions(run=demo["run"]), workers=2
    )
    report = read_report(receipt.output)
    assert receipt.status == report.status == "completed"
    assert len(report.cells) == 5
    assert all(cell.accuracy == 0.5 and not cell.pcmd.reportable for cell in report.cells)
    assert len(list(iter_results(receipt.output))) == 5


def test_detailed_result_corruption_is_detected(tmp_path):
    demo = json.loads(files("modebench").joinpath("walkthrough.json").read_text())
    source = tmp_path / "responses.jsonl"
    source.write_text(json.dumps(demo["records"][0]) + "\n")
    receipt = evaluate_file(source, tmp_path / "result.json")
    journal = receipt.work_dir / "results.jsonl"
    journal.write_text(journal.read_text() + "\n")
    with pytest.raises(InputError, match="hash mismatch"):
        read_report(receipt.output)
    with pytest.raises(InputError, match="hash mismatch"):
        list(iter_results(receipt.output))


def test_cards_and_licenses_cover_every_frozen_split():
    cards = json.loads(files("modebench").joinpath("dataset_cards.json").read_text())["cards"]
    splits = json.loads(files("modebench").joinpath("frozen_splits.json").read_text())["splits"]
    assert len(cards) == 26
    by_config = {card["config"]: card for card in cards}
    for split in splits:
        card = by_config[split["config_name"]]
        assert card["license"] == "CC-BY-4.0"
        assert card["splits"][split["split"]]["parquet_sha256"] == split["parquet_sha256"]
        assert card["splits"][split["split"]]["rows"] == split["rows"]
        assert len(card["authors"]) == 5
        if split["level"] == 4:
            assert any("not admitted" in s for s in card["limitations"])
        if split["domain"] == "pantry_plan":
            assert card["upstream_license"] == "CC0-1.0"
    assert (
        files("modebench").joinpath("data-sources.json").read_bytes()
        == (ROOT / "provenance/licenses/data-sources.json").read_bytes()
    )
