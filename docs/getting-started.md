# Getting started

Run the examples from the repository root. A Git clone supplies `data/`, `evidence/`, and `ops/`; installing only a wheel supplies the Python package and CLI.

## Installation

Follow the [README quick start](../README.md#quick-start). Check `python3 --version` first; if your system defaults to Python 3.9 or earlier, create the environment with an installed Python 3.10+ interpreter (for example, `python3.10 -m venv .venv`). The three install options are:

| Command | Use |
| --- | --- |
| `python -m pip install -e .` | Grade responses and calculate metrics |
| `python -m pip install -e '.[data]'` | Also load frozen Parquet splits and run construction tools |
| `python -m pip install -e '.[dev]'` | Also run the full regression suite |

To use the exact dependencies tested for this release on Linux/Python 3.10:

```sh
python -m pip install -c constraints-py310-tested.txt -e '.[dev]'
```

The [constraints file](../constraints-py310-tested.txt) records the clean CPU evaluation environment. It does not pin GPU training dependencies.

The core dependency is SymPy. The Python-factor validator executes restricted candidates in a separate, killable Python process. It is a task-specific interpreter, not a general-purpose sandbox for arbitrary programs. No model credentials or GPU are needed for these examples.

## 1. Check the data

```sh
python ops/verify_data.py
```

Expected summary: 72 verified splits and 15,552 rows. This checks frozen file identities without loading a model.

## 2. Prepare model requests

This example writes the 128 Level 1 Countdown evaluation requests. The `answer` reference remains in the local dataset and is never sent in a model request.

```python
import json
from pathlib import Path
from modebench.data import load_split
from modebench.prompts import make_messages, profile_metadata, prompt_sha256

rows = load_split("data", "level1_countdown", "eval")
Path("outputs").mkdir(exist_ok=True)
with Path("outputs/countdown_requests.jsonl").open("x") as handle:
    for index, row in enumerate(rows):
        request = {
            "id": f"countdown-{index}",
            "messages": make_messages(1, "countdown", row),
            "prompt_sha256": prompt_sha256(1, "countdown", row),
            "prompt_profile": profile_metadata(1, "countdown"),
        }
        handle.write(json.dumps(request) + "\n")
```

Send each `messages` list to your chosen model using one fixed generation protocol. Preserve response text, model/version, temperature, top-p, token limit, seeds where available, and all failures. Do not include `answer`, mode counts, or certified solutions in the model input.

## 3. Save generations for evaluation

The CLI accepts one prompt per JSONL line. `responses` is the list of raw response strings generated for that prompt. For example, this runnable smoke test creates a single synthetic Countdown record:

```python
import json
from pathlib import Path

record = {
    "id": "countdown-demo",
    "level": 1,
    "domain": "countdown",
    "answer": {"verifier": "countdown", "numbers": [1, 2, 3], "target": 6},
    "responses": ["1 + 2 + 3", "1 * 2 * 3", "1 + 2", "bad"],
}
Path("outputs").mkdir(exist_ok=True)
with Path("outputs/countdown_demo.jsonl").open("x") as handle:
    handle.write(json.dumps(record) + "\n")
```

For a real benchmark run, use the exact `row["answer"]` loaded in step 2 and your saved model responses. Match responses by stable prompt ID, rather than relying on asynchronous completion order. The reference may be a JSON object or the JSON-encoded string stored in the dataset.

Do not score incomplete collections as complete runs. Resolve collection failures under a declared protocol and retain their provenance. A model's emitted refusal or malformed answer is a response and remains in the denominator.

## 4. Grade and read the result

```sh
modebench evaluate outputs/countdown_demo.jsonl --output outputs/countdown_demo_metrics.json
```

For this example: accuracy = 0.5, pass@4 = 1.0, distinct@4 = 2.0, and PCMD = 1.0. Only one prompt is eligible, so `pcmd.reportable` is false under the default threshold. PCMD conditions on successful responses; the two failures still count toward accuracy and the sampling budget.

Read `cells[*].pcmd.d_mode` together with `defined_prompts`, `prompts`, `support`, and `reportable`. The [evaluation guide](evaluation.md) explains why a computed number may still be unsuitable for reporting.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `modebench: command not found` | Activate the installation environment, or use `python -m modebench.cli` with the same arguments. |
| `No module named datasets` | Install the `data` or `dev` extra. |
| Unknown dataset split | Use `eval`, not `test`. Consult the [configuration table](datasets.md#configuration-inventory); most Level 1 domains have no bundled `dev` split. |
| Dataset hash mismatch | Restore the frozen file from the same repository revision. Keep newly generated variants under a new config and output directory. |
| Output file already exists | Choose a new output path. The CLI and materializer intentionally refuse overwrites. |
| PCMD is `null` or unreportable | Check the number of verified responses per prompt and the eligible-prompt count; do not replace missing PCMD with zero. |
| Python-factor execution fails on Windows | Use a POSIX environment, such as WSL. |
| Pantry support mask is rejected by a low-level validator | Use `grade_response` or the CLI, which apply the Level 1 support-mask interface. |
