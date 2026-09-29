# ModeBench

**Can a model find more than one correct solution to the same problem?** ModeBench evaluates correctness and solution diversity using executable tasks with canonical outcome identities.

[![The five ModeBench domains, each showing two verified solution modes.](assets/modebench-domains.png)](assets/modebench-domains.png)

Five domains, five task levels, and **72 frozen dataset splits containing 15,552 rows**. Evaluate saved model responses on CPU without PyTorch, CUDA, or a training framework. [Re:Max / Re:Dr](https://github.com/liv-daliberti/remax) provides the separate optimization and training implementation.

| Domain | What is checked | What defines a mode |
| --- | --- | --- |
| Graph coloring | Every graph edge and fixed color | Complete color vector |
| Countdown | Exact arithmetic and use of the given operands | Normalized executed expression tree |
| Python factors | A restricted function returns proper divisors on the given inputs | Returned divisor vector |
| MathIR | Restricted equation actions execute correctly | Exact normalized state trajectory |
| PantryPlan | Ingredient and nutritional feasibility constraints | Ingredient support |

Mode identity is prompt-local. Different wording alone does not create a different mode; different Python programs can produce the same divisor vector.

[Installation](#installation) · [Evaluation](#evaluation) · [Metrics](#metrics) · [Datasets](#datasets) · [Reproducibility](#reproducibility) · [Contributing](#contributing) · [Changelog](#changelog) · [Citation](#citation)

## Installation

The supported runtime is **Linux, CPython 3.10–3.12**. Windows users can use a Linux environment under WSL. Native Windows and macOS are not qualified: verifier deadlines, process cleanup, and memory limits use Linux/POSIX facilities. Evaluation is CPU-only.

Install a regular package, then run the complete offline walkthrough:

```sh
git clone https://github.com/liv-daliberti/modeBench.git
cd modeBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
modebench walkthrough --directory demo
head -n 1 demo/tasks.jsonl
modebench evaluate demo/responses.jsonl --run demo/run.json --output demo/report.json
modebench report demo/report.json
```

The walkthrough contains **one task from each of the five domains**, four saved responses per task, run metadata, and `expected.json`. Inspect `tasks.jsonl` for the questions; edit a copy of `responses.jsonl` to supply your own saved responses. These are handwritten examples, with two correct modes, one wrong answer, and one malformed answer per task. No model calls or downloads occur, and the core package works outside the checkout.

The report prints one row per domain with these values:

```text
level domain          k accuracy pass@k distinct@k PCMD reportable
1     countdown       4 0.5      1      2          1    false
```

All five rows have the same scores. `reportable=false` is expected: one eligible prompt is below the default threshold of 30. This walkthrough demonstrates the interface; it is a custom-reference example, not a full benchmark result. Choose a fresh output filename for each run; existing outputs and walkthrough directories are never overwritten.

| Installation from checkout | Includes |
| --- | --- |
| `python -m pip install .` | All five verifiers, metrics, CLI, frozen split registry, and offline walkthrough |
| `python -m pip install '.[data]'` | Also Parquet loading and dataset construction dependencies |
| `python -m pip install '.[dev]'` | Also the regression suite |

The wheel and source distribution contain the same runtime resources; the checkout additionally supplies frozen Parquet files, scientific evidence, and construction scripts. CI uploads both distributions as the `distributions` artifact. They can be installed directly with `pip install /path/to/package.whl` or `pip install /path/to/package.tar.gz`; add `[data]` to that path for dataset loading. PyPI publication is not configured yet; do not assume an unrelated package with the same name is this release.

Supported direct dependency ranges are SymPy `>=1.12,<2`, and, for data, datasets `>=2.16.1,<6` and huggingface-hub `>=0.19.4,<2`. CI checks the minimum combination using [compatibility constraints](provenance/constraints-minimum.txt), and current resolver-selected versions. When using datasets 2.16, use those constraints: its older Arrow/NumPy integration requires them. This tests the endpoints, not every possible dependency combination. The [recorded Python 3.10 environment](provenance/constraints-py310.txt) is another reproducible installation choice.

### Discover and inspect real tasks

```sh
python -m pip install '.[data]'
modebench datasets list
modebench datasets card --config level1_countdown
modebench datasets fetch --config level1_countdown
modebench datasets inspect --config level1_countdown --row 0 --offline
```

`list` prints 72 split records ordered by level and domain; `--json` gives their hashes and row counts. `fetch` prints `verified: true` and the cache data directory. `inspect` shows ID `level1_countdown/eval/0`, 128 total rows, the dataset hash, and registered messages. Answers are omitted unless you explicitly use `--show-reference`; never send that inspection output to a model.

Downloads use a pinned public Git commit and must match SHA-256 identities shipped in the package. The cache defaults to `$XDG_CACHE_HOME/modebench` (or `~/.cache/modebench`), partitioned by registry hash. Every reuse verifies the bytes. `--cache-dir PATH` selects a different base, and `--offline` forbids downloads. Failed or corrupt downloads never become usable cache entries. A corrupt existing entry produces an error with its path; remove that file explicitly and fetch it again. Transfers have a 32 MiB cap and bounded network waits.

`prepare`, `datasets inspect`, and frozen `evaluate` use `--data-root` when supplied, otherwise the checkout's `data/` if present, otherwise the verified cache (downloading a missing split unless `--offline`). Use `--cache-dir` to select the cache explicitly even inside a checkout. Dataset discovery and downloading need only the core package; reading Parquet needs the `data` extra.

## Evaluation

Use the typed public API to load a frozen task, render its prompt, and grade a saved response:

```python
from modebench.api import load_tasks, make_prompt, grade

task = next(load_tasks("level1_countdown", data_root="data"))
messages = make_prompt(task)
# Send only `messages` to your model, then grade its returned text.
result = grade(task, "YOUR_MODEL_RESPONSE")
print(result["status"], result["canonical_key"], result["detail"])
```

Outside a checkout, omit `data_root` to use the verified cache; `cache_dir` and `offline=True` are supported. Reading frozen tasks requires the `data` extra. `Task` contains the reference for grading; only `make_prompt(task)` belongs in a model request.

For saved response files, the same API exposes typed evaluation options, receipts, reports, and streaming details:

```python
from modebench.api import EvaluationOptions, evaluate_file, read_report, iter_results

receipt = evaluate_file(
    "responses.jsonl", "results.json",
    options=EvaluationOptions(min_defined_prompts=30),
)
report = read_report(receipt.output)
print(report.status, report.cells)
for prompt in iter_results(receipt.output):
    print(prompt["id"], prompt["accuracy"])
```

`read_report` verifies the detailed-result journal hash. `iter_results` verifies it before yielding results in input order. Both read v3 and v4 receipts. Type information ships in the package (`py.typed`); new integrations should use `modebench.api` instead of private helpers or historical adapters.

Keep `answer`, support counts, and certified solutions out of model requests. Match saved responses by stable prompt ID, not asynchronous completion order. Preserve raw generations, refusals, and malformed answers; collection failures need a declared protocol and must not silently reduce the draw budget.

**Custom-reference evaluation:** write one JSON object per prompt to a JSONL file:

```json
{"id":"example","level":1,"domain":"countdown","answer":{"verifier":"countdown","numbers":[1,2,3],"target":6},"responses":["1+2+3","1*2*3","1+2","bad"]}
```

```sh
modebench evaluate INPUT.jsonl --output OUTPUT.json
```

| Field | Contract |
| --- | --- |
| `id` | Stable prompt identity; string recommended, unique within level/domain |
| `level` | Integer 1–5 or `"level1"` through `"level5"` |
| `domain` | `graph_coloring`, `countdown`, `python_factors`, `mathir`, or `pantry_plan` (`pantry` is an alias) |
| `answer` | Exact executable reference from the frozen row, as an object or JSON-encoded string |
| `responses` | Nonempty list of raw response strings, including failed answers |

Use one input file per model and generation condition. Grouping uses level, domain, and response count `k`. The evaluator rejects unknown record fields, malformed references, domain/verifier mismatches, duplicate JSON fields, and nonfinite numbers before grading. CLI errors identify the input line and prompt. Integer counts and positive integer thresholds are enforced by the metric API too.

Custom references are labeled `dataset.kind: "custom"` and `references_authenticated: false`. Use `--run RUN.json` to record model, generation, and prompt metadata; without it, those details are explicitly unreported. An inline `run` object on every record is also supported. Inline metadata must match the shared run exactly; mixed models, revisions, generation settings, or prompt conditions are rejected. Arbitrary record fields such as `model_id` are rejected rather than silently ignored.

File evaluations use `modebench-saved-responses-v4`: a compact final receipt contains the input SHA-256, software identity, run metadata, dataset identity, aggregation settings, and cell scores. Per-response diagnostics live in a separate JSONL journal whose path, SHA-256, and record count are bound into the receipt. Unverified responses have a null canonical key; evaluator failures also invalidate aggregate scores, as described below. For Level 1 Pantry, `graded_text` is the deterministic allocation projected from the submitted six-bit support mask. Use `modebench.api.grade` or the CLI for this interface; the low-level key validator does not apply that projection.

### Streaming, interruption, and parallel workers

```sh
modebench evaluate responses.jsonl --run run.json --output results.json \
  --progress-every 100
# After an interrupted run, repeat the same command with --resume:
modebench evaluate responses.jsonl --run run.json --output results.json --resume
modebench report results.json
```

Input is streamed one JSONL record at a time. All records and references are validated before any candidate grading. Duplicate IDs are checked on disk. Each completed prompt is committed to a SQLite checkpoint and appended to `results.json.work/results.jsonl`; progress events go to stderr and the final summary goes to stdout. `--progress-every 0` disables progress. Final receipts are published atomically and never overwrite an existing output. A partial journal alone is not a completed benchmark result.

The default checkpoint directory is `OUTPUT.work/`; `--work-dir PATH` selects another fresh directory. Resume verifies the input bytes and records, options, run metadata, dataset identities, software/dependency fingerprints, resource policy, and stored prompt/result hashes. It rebuilds a damaged or missing journal from committed checkpoints, then continues without regrading completed prompts. An interruption during input validation restarts validation. Changed responses, configuration, references, or software require a fresh evaluation. A completed evaluator failure is retained on resume; resume is not a selective retry policy.

`--workers N` supports 1–8 isolated verifier workers, with bounded pending work and ordered results. The default is **1**. Worker count can change on resume; execution history records it. Tests compare serial, interrupted/resumed, and parallel scores and detailed results, including abrupt process termination and worker cleanup. Concurrency consumes additional memory and may not improve throughput on every machine.

Retain the receipt and its journal together, preserving the relative path in `prompt_results.path`. The SQLite ledger is needed to resume, but not to read a completed report. Checkpoints contain raw responses and references: allocate disk space and handle them like the original inputs. Durability relies on filesystem locking and `fsync`; use a local filesystem with those guarantees. An existing final receipt means publication completed: read it or choose a fresh output rather than using `--resume` to overwrite it. Keep input files immutable during a run. Input records are capped at 16 MiB and 4,096 responses per prompt; worker limits below still apply.

Measured on a shared Linux host with 10,000 repeated walkthrough prompts (40,000 responses):

| Execution | Parent peak RSS | Parent + workers peak RSS | Responses/second |
| --- | ---: | ---: | ---: |
| Previous in-memory CLI | 292.5 MB | 339.2 MB | 312.8 |
| Streaming, one worker | 60.1 MB | 106.9 MB | 219.1 |
| Streaming, two workers | 61.2 MB | 152.1 MB | 295.3 |

Serial streaming parent memory was 59.1 MB at 1,000 prompts. Durable per-prompt commits trade throughput and disk space for recovery; the two-worker 10,000-prompt checkpoint occupied about 109 MB, including its 18 MB journal. These are single development measurements with repeated tasks and warm caches, not general performance guarantees. Process-tree RSS can double-count shared pages. [Measurement records](provenance/performance.json) preserve the methods and limitations. Reproduce on your own storage and workload before selecting concurrency:

```sh
python ops/benchmark_evaluation.py --sizes 1000 10000 --workers 1 2 \
  --output outputs/performance.json
```

### Evaluate a frozen split

Export registered requests with stable IDs and prompt/dataset hashes. Reference answers and support counts are never included:

```sh
modebench prepare --config level1_countdown --output outputs/requests.jsonl
```

Send each `messages` list to your model. Save the exported records to a response file with a `responses` list added to each record. Preserve IDs and hashes. IDs have the form `level1_countdown/eval/0`: configuration, split, and zero-based row index. They are stable within the frozen split identity.

Copy [examples/run.json](examples/run.json) to your run directory and replace its illustrative values with the actual settings:

```json
{
  "model": {"id": "your-model", "revision": "exact-model-revision"},
  "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 192, "seed": 43},
  "prompt_condition": "registered_hints_v1"
}
```

Then evaluate against the frozen references:

```sh
modebench evaluate outputs/responses.jsonl --config level1_countdown \
  --run outputs/run.json --output outputs/metrics.json
```

`--split` defaults to `eval`; `train` and `dev` are also accepted where available. `--data-root` selects an explicit directory; otherwise the checkout or verified cache is used. Frozen evaluation compares the local manifest against the registry shipped inside the package, verifies Parquet bytes, resolves each prompt ID, and supplies its reference internally. Do not include `answer` in frozen response records. Optional supplied `level`, `domain`, `problem`, `messages`, `prompt_sha256`, and `dataset_sha256` must match the frozen source and registered prompt.

Frozen records must have the same draw count per prompt; failures remain in the response lists. An optional `generation.draws_per_prompt` in run metadata is checked against every record in either mode. Complete split coverage is required by default. For an intentional subset, use `--allow-partial`; results list selected and missing IDs, counts, and `complete: false`. Missing prompts are not imputed. Frozen evaluation requires `registered_hints_v1`; changed prompt conditions belong in custom mode. Frozen reference binding does not mean every dataset is an admitted benchmark condition: Level 4 and diagnostic qualifications remain attached to the results.

<details>
<summary>Result identities and compatibility</summary>

| Field | Meaning |
| --- | --- |
| `software` | ModeBench version, verifier contract, package-source hashes, Python/dependency versions, Git commit/dirty state when available |
| `input_sha256`, `records_sha256` | File bytes and canonical JSON records; the in-memory API supplies only the record digest |
| `run`, `run_sha256` | Declared model/revision, generation settings, prompt condition, and metadata digest |
| `dataset` | Custom versus frozen binding; frozen split metadata, hashes, coverage, and missing IDs |
| `evaluation` | Aggregation, equal prompt weighting, support threshold, and empirical pass@k convention |
| `cells[*].protocol_notes` | Level 4 admission and MathIR matching caveats |
| `prompt_results` | In v4, path/hash/count of the detailed JSONL journal; each entry contains prompt/reference hashes, prompt profile, scores, and graded attempts |

Generation settings are **user-declared**, not proof that a model produced the responses. Prompt hashes identify the expected registered messages, not the actual transport used. Custom records need `problem` plus declared run metadata to compute a prompt hash; otherwise it is null. Model identity/revision strings and a nonempty generation-settings object are required whenever run metadata is supplied. Record any unavailable provider revision explicitly, with the collection date; do not invent one.

Version 0.4 keeps the existing `evaluate` command and in-memory `modebench.cli.evaluate` import. Accepted responses and canonical identities remain checked against the frozen baseline. The file receipt advances from v3 to v4 to separate aggregate metadata from detailed results; the in-memory API still returns nested v3 results. Use `read_report` and `iter_results` for schema-independent consumers. Consumers must check `evaluation.status` before reading scores. Legacy key-only Python verification still returns `None` for candidate mistakes but raises `VerifierExecutionError` for execution failures.

The Python API also supports `evaluate(records, run=metadata, data_root="data", config="level1_countdown", split="eval", allow_partial=False)` through `modebench.evaluation`.

</details>

### Wrong answers versus evaluator failures

Every public `grade` / `grade_response` result includes a `status`, `verified`, `canonical_key`, `graded_text`, and diagnostic `detail`:

| Status | Meaning | Registered scoring policy |
| --- | --- | --- |
| `correct` | The answer verifies | Count success and its canonical mode |
| `incorrect` | Allowed answer syntax, wrong output or violated task constraints | Count a failed draw |
| `malformed` | Unparseable answer or disallowed candidate syntax | Count a failed draw |
| `timeout` | Worker execution or parent request deadline exceeded | Fail the evaluation; suppress all aggregate scores |
| `invalid_reference` | Invalid task/reference, not a model mistake | Reject input; never count a failed draw |
| `worker_failure` | Worker crashed, bad protocol reply, startup error, or unexpected backend exception | Fail the evaluation; suppress all aggregate scores |
| `resource_limit` | Operational request/response or memory cap exceeded | Fail the evaluation; suppress all aggregate scores |

The policy identity is `wrong-or-malformed-count-as-failure; verifier-errors-suppress-all-aggregates-v1`. A run containing any unscorable attempt has `evaluation.status: "failed"`, null accuracy/pass/distinct/PCMD aggregates, and `reportable: false`. Attempt diagnostics remain in the result journal (nested in legacy v3 receipts); healthy prompts are not silently substituted for the failed run. The metric API rejects attempts with evaluator-failure statuses too. Invalid references are normally rejected during input preflight, before any responses are graded.

CLI exit codes are **0** for completed evaluation, **2** for input/usage errors, and **3** for evaluator failure. Execution failures write a diagnostic receipt, including failures during reference validation. `modebench report` also exits 3 for a failed receipt. Fix the evaluation environment or reference and rerun under a recorded protocol; do not replace the failure with an incorrect answer or silently retry only that draw.

All five domains and reference validation run in isolated worker processes. Python candidate execution has a 0.25-second wall deadline; other operations have 2 seconds. The parent bounds startup and pipe I/O with a 5-second request deadline and reaps failed workers; the next request can start a new worker. Workers have a 1 GiB address-space cap. These are operational limits, not promises about throughput or a general-purpose sandbox for arbitrary programs.

| Boundary | Limit |
| --- | --- |
| Raw response / serialized reference | 131,072 characters / 64 KiB |
| Worker request / reply | 1 MiB / 256 KiB |
| Countdown and graph candidate extraction | 160 characters |
| Python candidate | 240 characters, 64 AST nodes; up to 8 reference inputs, each at most 1,000 |
| MathIR candidate | 160 characters, at most 4 steps; expression tree depth/node bounds |
| Pantry candidate | 512 characters, at most 16 ingredients; bounded quantities and isolated support projection |

Candidate grammar limits preserve the existing benchmark syntax and normally produce `malformed`. Operational exhaustion is unscorable. The [resource audit](provenance/verifier-resource-audit.json) records observed reference sizes and domain bounds across all 15,552 frozen rows. Tests cover deadline interruption, partial and blocked pipe I/O, memory exhaustion, worker termination, restart, cleanup, and unexpected backend failures in every domain.

<details>
<summary>Troubleshooting</summary>

| Symptom | Resolution |
| --- | --- |
| CLI not found | Activate the environment or run `python -m modebench.cli`. |
| Missing `datasets` | Install the `data` or `dev` extra. |
| Unknown split | Use `eval`, not `test`; check split availability below. |
| Dataset hash mismatch | Restore the file from the same revision; keep variants under separate identities. |
| Output already exists | Read the completed report or choose a fresh destination. Resume only interrupted runs without a final receipt. |
| Checkpoint exists / identity mismatch | Use `--resume` with the original input and configuration, or choose a fresh destination for a changed experiment. |
| Detailed-result hash mismatch | Restore the original journal; an interrupted run can rebuild it from its checkpoint with `--resume`. |
| PCMD is null or unreportable | Check verified draws and eligible-prompt support; missing PCMD is not zero. |
| CLI exits 3 / evaluator failure | Inspect `failure` or attempt `status`/`detail`; fix the cause and rerun without treating it as model failure. |
| Native Windows or macOS worker errors | Use the supported Linux runtime; WSL provides one on Windows. |
| Offline cache miss | Fetch the same configuration/split once without `--offline`, then retry. |
| Changed response count / missing prompts | Preserve failed draws; use `--allow-partial` only for an explicitly declared subset. |

</details>

## Metrics

For a prompt with `k` draws, let `K` verify and let `n_m` count verified responses in mode `m`.

| Metric | Per-prompt definition | Aggregation |
| --- | --- | --- |
| Accuracy | `K / k` | Equal mean over prompts |
| Empirical pass@k | `1` if any response verifies, otherwise `0` | Equal mean over prompts |
| Distinct@k | Number of distinct verified modes | Equal mean over prompts |
| PCMD | `1 - sum(n_m * (n_m - 1)) / (K * (K - 1))` | Equal mean over prompts with `K >= 2` |

PCMD measures success-conditional diversity. Counts `{A: 2, B: 1}` give `2/3`; one repeated mode gives zero. Fewer than two successes gives `null`, not zero. Eligibility depends on correctness, so always report accuracy, `defined_prompts`, `prompts`, `support`, and `reportable` alongside PCMD.

The default reporting threshold is 30 eligible prompts. A value may exist with `reportable: false`; respect that flag in tables and figures. `--min-defined-prompts` changes the threshold and should be registered before analysis. Empirical pass@k describes the saved groups, not the combinatorial estimator for subsampling a larger pool.

<details>
<summary>Repeated groups and uncertainty</summary>

The CLI treats each prompt's response list as one group. The Python metric API supports repeated groups:

```python
from modebench.metrics import POOLED, PER_GROUP, prompt_mode_diversity

draws = [{"attempts": [
    {"verified": True, "canonical_key": "A"},
    {"verified": True, "canonical_key": "B"},
    {"verified": False, "canonical_key": None},
]}]
assert prompt_mode_diversity(draws, POOLED) == 1.0
assert prompt_mode_diversity(draws, PER_GROUP) == 1.0
```

`POOLED` pools a prompt's groups; `PER_GROUP` averages its defined group scores. Do not flatten four groups of eight and label the resulting pass@32 as pass@8. Never pool mode counts across prompts.

`standard_error` is the sample standard deviation of defined prompt scores divided by the square root of their count. It is null for one defined prompt and is not a paired comparison interval. Finite-sample eligibility changes the population being compared.

`rarefied_distinct_at_2` equals `1 + PCMD`. `effective_modes` is `1 / (1 - mean_PCMD)`, computed after aggregation; it is null when PCMD is undefined or equals one, and is not an unbiased mode-count estimate.

</details>

## Datasets

Files are organized as [`data/<level>/<domain>/<split>.parquet`](data/). Configuration names remain `level1_countdown`, `level2_mathir`, and so on. The [manifest](data/manifest.json) records every configuration, split, row count, original subset, and hash. `load_split` verifies the Parquet SHA-256 and row count. The public `load_tasks` API handles verified downloads and caching outside the checkout.

| Configurations | Train | Dev | Eval |
| --- | ---: | ---: | ---: |
| Level 1 Countdown, graph coloring, MathIR, Python factors | 384 | — | 128 |
| Level 1 PantryPlan | 384 | 64 | 128 |
| Levels 2–5, all five domains | 384 | 128 | 128 |
| Level 1 graph-coloring single-answer diagnostic | — | — | 128 |

`modebench datasets card --config CONFIG` displays the packaged dataset card: task and mode definitions, fields, split hashes/counts, authors, source terms, intended use, and limitations. Cards cover all 25 cells and the separate single-answer diagnostic.

A dash means absent. The diagnostic lives under `data/level1/graph_coloring/unique_answer/` and uses configuration `level1_graph_coloring_unique_answer`; it is separate from the main benchmark.

Rows contain `problem`, `answer`, `modebench_task`, `answer_mode_count`, and `answer_mode_split`, plus domain-specific metadata. Python factors expects one restricted `lambda n: EXPR`; MathIR uses equation actions/menu selections; Level 1 Pantry uses a six-bit support mask with deterministic quantity projection. Other Pantry levels use allocation responses.

Prompts default to `registered_hints_v1`, matching the bundled data. The optional `python_level3_neutral_v1` changes Level 3 Python wording; its separately calibrated dataset is not included. Switching conditions changes the experiment. Structured chat messages do not reproduce training-time syntax masks; record free versus constrained decoding.

**Level 4 MathIR is not difficulty-matched, and Level 4 as a whole is not admitted.** The other four Level 4 domains passed their matching checks. All five Level 5 domains are admitted according to the retained manifests. Admission reflects a recorded calibration procedure, not a theorem of equal difficulty. See the [Levels 1–3](provenance/levels123_manifest.json) and [Levels 4–5](provenance/levels45_manifest.json) source records.

To materialize a frozen configuration for the separate training repository:

```sh
python ops/materialize_training_data.py --config level1_countdown --output outputs/countdown
```

This creates `train/` with subset `train` and `eval/` with subset `multi_answer`, preserving frozen rows and refusing an existing destination. Basic-domain generators remain in [ops/](ops/); consult their `--help`. Pantry construction requires an explicit ingredient-table JSON via `--ingredients`. Full higher-level construction, fitting, and admission workflows are not consolidated; generated variants require separate identities and admission evidence.

## Reproducibility

```sh
make check                          # Tests, conformance, analysis, data hashes, and scoped quality checks
python ops/verify_release.py         # Repository file inventory and hashes
```

Validation covers the regression suite, **175 frozen response cases across all 25 level/domain cells**, **375 frozen base-grid cells**, and **72 splits / 15,552 rows**. The conformance corpus includes accepted/rejected responses, formatting boundaries, and equal/different canonical identities. Exact arithmetic and separately implemented domain checks review the accepted witnesses and mode distinctions independently of production validators. This is automated independent checking; an external human review is still outstanding.

Fixtures freeze behavior from commit `0ae27d301c144411d6a3b67b9a3d2483d33b02bb`, with a locked corpus hash. CI additionally compares the new verifier against the PR base's fixture bytes, so changing expected values alongside a refactor does not erase the baseline. Metric properties cover bounds, permutations, renamed modes, and exact pair-count identities; canonicalization properties and deliberate semantic mutations check that the suite detects drift. Finite fixtures cannot prove equivalence for every possible input.

CI builds a wheel **from the sdist**, validates package metadata, and runs the regression matrix against regular installs on Python 3.10, 3.11, and 3.12. Six isolated installation combinations cover wheel/sdist, core/data extras, and minimum/current dependencies. Their smoke script runs outside the checkout with no source-path overrides, verifies the import comes from site-packages and is not editable, executes all 175 conformance cases and the five-domain walkthrough, checks typed API resources, dataset cards, resume and parallel equivalence, and exercises offline frozen prepare/inspect/evaluate with dataset extras. The suite does not require live network access to dataset hosting.

To reproduce an installation check locally:

```sh
python -m pip install build
python -m build
python ops/check_install.py --artifact dist/modebench-0.4.0-py3-none-any.whl
python ops/check_install.py --artifact dist/modebench-0.4.0.tar.gz --profile data --minimum
```

This creates and removes a fresh virtual environment and an external working directory for each check. Installing dependencies requires network access or a configured package mirror. The minimum data stack uses the compatibility constraints described above. Versioned PyPI packages and separately hosted dataset releases remain a release-management follow-up; the current dataset URL is already pinned to immutable Git bytes. Future publishing should use [PyPA's trusted-publisher workflow](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) after repository ownership and publisher configuration are established.

The [verified-key archive](evidence/base_grid_keys.jsonl.gz) retains attempt flags/keys, repeated groups, recorded pass@8/distinct@8, and source receipt hashes. `ops/reproduce_base_grid.py` compares its recomputed summaries with the [frozen results](evidence/mode_diversity_base_grid.json), allowing only final-bit numeric tolerance (`rel_tol=1e-14`, `abs_tol=1e-15`); identifiers and counts match exactly. The archive lacks complete raw responses, so this check does not independently regrade generations or reproduce model sampling.

[Source provenance](provenance/source.json) records extracted-file identities; the [release manifest](provenance/release.json) binds final files. Extraction included uncommitted research changes, so the source commit alone is insufficient. Historical machine paths identify source artifacts and are not runtime dependencies. Equivalence to every historical verifier runtime remains unestablished; fresh model generation and training-checkpoint reproduction are outside these checks.

For reported results, retain raw responses and record the repository commit, dataset configuration/split/hash, selected prompt IDs and exclusions, prompt condition/hash, model revision, decoding settings, seeds/dates, draw/group structure, failures, metric support/threshold, and uncertainty method. Use the repository URL and exact commit in software citations; use the [paper citation](#citation) alongside those software identities.

### Repository layout

```text
src/modebench/
  domains/
    countdown/       # arithmetic, expression identities, typed grading
    graph_coloring/  # complete/partial colorings, edge constraints
    python_factors/  # restricted Python, divisor vectors, process adapter
    mathir/          # symbolic equation actions and state trajectories
    pantry_plan/     # allocations, feasibility, support-mask projection
  api.py, api_types.py                  # supported typed public interface
  cli.py, evaluation.py, metrics.py      # shared evaluation contract
  streaming.py, parallel.py, reporting.py # durable file evaluation and reports
  data.py, dataset_cache.py             # shared data access
  verifier.py, verifier_worker.py       # public diagnostics and worker dispatch
  worker_process.py, diagnostics.py     # process lifecycle and failure policy
```

Each domain owns its reference validation, answer validation, and canonical identities. Prompt profiles and common response extraction remain shared. Earlier imports such as `modebench.mathir` and `modebench.python_modebench` resolve to the corresponding domain modules for compatibility; new integrations should use `modebench.api`, with domain implementation code under `modebench.domains.<domain>`. The old `historical_prompts` import aliases `registered_prompts`. Software fingerprints include nested domain files.

Keep `evidence/` and `provenance/`: they serve different reproducibility needs. **Evidence** holds the frozen verified-key archive and numerical results used to detect metric drift. **Provenance** holds dataset origins, attribution, admission records, dependency constraints, the resource audit, and release-file identities. Neither is required to grade a saved response in an installed package, but both belong in the scientific repository. The wheel carries the runtime registry, walkthrough, dataset cards and attribution, type metadata, and applicable license notices.

## Contributing

Liv G. d’Aliberti (`@liv-daliberti`) is the repository maintenance and scientific-review contact, as recorded in [CODEOWNERS](.github/CODEOWNERS). The five paper authors are credited in [CITATION.cff](CITATION.cff); this does not make every author a software support contact.

Run `make check` before submitting changes. Report bugs through [GitHub issues](https://github.com/liv-daliberti/modeBench/issues) with the commit, Python version, domain/level, and a minimal response/reference example.

Explain changes to accepted responses, canonical identities, prompts, splits, or aggregation, and add focused behavioral tests. The packaged `src/modebench/frozen_splits.json` registry must match `data/manifest.json`; adding a frozen release is an explicit scientific version change, not a hash repair. Keep frozen datasets and evidence immutable; changed conditions need new identities. Preserve the worker execution boundary for all five domains and keep the core independent of training frameworks. Do not regenerate conformance expected values to make a refactor pass: explain intentional benchmark changes and version their scientific contract explicitly.

After reviewing deliberate file changes, refresh the release inventory with `python ops/verify_release.py --refresh`, inspect its diff, then verify it again. Updating hashes records changed bytes; it does not establish scientific equivalence.

### Software fixes versus benchmark changes

A **software fix** preserves accepted/rejected responses, canonical identities, registered prompt bytes, dataset hashes, scoring definitions, and admitted conditions. Packaging, diagnostics, performance, recovery, and compatible APIs can receive a software release after conformance and frozen numerical checks pass. Changes to output schemas must be documented and retain explicit schema identities.

A **benchmark change** alters any of those scientific contracts, including a verifier bug fix that changes which answers score. It requires a new benchmark/condition identity, explicit before/after evidence, versioned fixtures and dataset identities when applicable, scientific review coordinated by the maintainer, and a changelog explanation of which results remain comparable. Never overwrite old datasets or silently refresh reference expectations. A package version alone does not make changed scientific results comparable. Resource or failure-policy changes must be recorded and assessed for changed scoreability.

Formatting and linting are enforced on the new public API, evaluation/recovery/reporting code, and their focused tests. Strict type checks cover `api.py`, `api_types.py`, and `contracts.py`, with positive and deliberately invalid client examples. `make quality` also validates citation metadata. Legacy domain internals are not yet fully typed or uniformly formatted; expand the checked scope when touching them, with semantic conformance checks alongside changes.

## Changelog

- **0.4.0** — Streamed file evaluation with durable per-prompt checkpoints, input/configuration-bound resume, progress, atomic v4 receipts, hash-bound result journals, and optional bounded parallel workers. Added a typed public API, dataset cards, scoped quality checks, citation metadata, ownership/version policies, and CC BY 4.0 licensing for the authors' dataset contributions. In-memory v3 APIs and legacy module aliases remain available; registered benchmark meaning and frozen dataset bytes are unchanged.
- **0.3.0** — Structured verifier failures, isolated resource-bounded verification, frozen conformance coverage, domain packages, verified dataset discovery/cache, five-domain walkthrough, and clean wheel/sdist installation testing.


## Citation

**Measuring and Mitigating Solution Mode Collapse in RLVR** has been accepted at the [6th Workshop on Mathematical Reasoning and AI (MATH-AI), NeurIPS 2026](https://mathai-2026.github.io/), and submitted to **ICLR 2027, where it is under review**.

```bibtex
@inproceedings{dAliberti:etal:ModeCollapse:2027,
  author    = {d'Aliberti, Liv G. and Abdulhai, Marwa and Druchyna, Sofiia and Henderson, Peter and Horta Ribeiro, Manoel},
  title     = {Measuring and Mitigating Solution Mode Collapse in {RLVR}},
  booktitle = {International Conference on Learning Representations ({ICLR})},
  year      = {2027},
  note      = {Under review at {ICLR} 2027},
}
```

[CITATION.cff](CITATION.cff) supplies machine-readable citation metadata. Alongside the paper citation, record the ModeBench software version and commit, dataset configuration/split/hash, prompt condition, and verifier contract. The ICLR entry records the submission; it does not claim ICLR acceptance.

## License and attribution

Code is licensed under [Apache 2.0](LICENSE), with original notices retained. The authors' copyrightable dataset contributions are licensed under **[CC BY 4.0](DATA_LICENSE)** on behalf of Liv G. d'Aliberti, Marwa Abdulhai, Sofiia Druchyna, Peter Henderson, and Manoel Horta Ribeiro. Credit the authors and ModeBench, link the license, and indicate changes. The grant covers the frozen task text, executable references, annotations, compilation, and their copies in walkthroughs and fixtures.

PantryPlan's upstream USDA FoodData Central records remain **CC0/public domain** under the [USDA terms](https://fdc.nal.usda.gov/api-guide/); the dataset grant adds no conditions to those elements. [Source attribution and scope](provenance/licenses/data-sources.json) distinguish project contributions from upstream material. The distribution's combined license expression reflects its software and data components. Model weights, unrelated third-party software, and the manuscript are outside this grant. Historical provenance statements recording an unresolved dataset license describe the earlier export; the current [dataset license](DATA_LICENSE) supersedes that project-level status.
