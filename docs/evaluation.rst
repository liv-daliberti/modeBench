Evaluate, interrupt, and resume
===============================

Evaluate a frozen split
-----------------------

After :doc:`datasets`, evaluate the completed response records:

.. code-block:: console

   modebench evaluate responses.jsonl --config level1_countdown \
     --run run.json --output results.json --progress-every 100
   modebench report results.json

The evaluator supplies references from the authenticated dataset. Do not include
``answer`` in frozen response records. IDs, supplied messages/hashes, draw counts,
and the registered prompt condition must match. The default split is ``eval``,
not ``test``. Full split coverage is required by default; ``--allow-partial`` labels
an intentional subset with its missing IDs rather than imputing missing results.

Use one input file per model and generation condition. Existing output files are
never overwritten. Unknown fields, duplicate JSON keys, nonfinite numbers,
malformed references, and mixed run identities fail validation.

Custom-reference inputs
-----------------------

For a small diagnostic, save this as one line in ``custom.jsonl``:

.. code-block:: json

   {"id":"example","level":1,"domain":"countdown","answer":{"verifier":"countdown","numbers":[1,2,3],"target":6},"responses":["1+2+3","1*2*3","1+2","bad"]}

.. code-block:: console

   modebench evaluate custom.jsonl --output custom-report.json

Custom references are labeled ``dataset.kind: "custom"`` and are not authenticated
frozen benchmark references. Without run metadata, model and generation settings
are explicitly unreported.

Wrong answers and evaluator failures
------------------------------------

.. list-table::
   :header-rows: 1

   * - Status
     - Interpretation
   * - ``correct``
     - Verified response with a canonical key.
   * - ``incorrect``
     - A scorable response that does not solve the task.
   * - ``malformed``
     - Candidate syntax or formatting violates the registered contract.
   * - ``timeout``
     - Evaluation exceeded its operational deadline.
   * - ``invalid_reference``
     - The grading reference is not valid.
   * - ``worker_failure``
     - The evaluator process or communication failed.
   * - ``resource_limit``
     - Evaluation exhausted an operational resource bound.

The last four are evaluator failures, not zero-reward answers. They invalidate
aggregate scores; the CLI reports failure and exits with status 3. Inspect each
attempt's ``status`` and ``detail``. Fix the cause and run a fresh evaluation with
an explicit retry policy; do not selectively discard failed attempts.

Streaming and recovery
----------------------

Input is streamed one prompt at a time. Before grading starts, all records and
references are validated. Completed prompts are committed to a SQLite ledger in
``results.json.work/`` and appended to a detailed JSONL journal. Progress goes to
stderr; the final receipt is published atomically.

After an interruption, repeat the original command with ``--resume``:

.. code-block:: console

   modebench evaluate responses.jsonl --config level1_countdown \
     --run run.json --output results.json --resume

Resume authenticates the input, configuration, dataset, installed software,
dependencies, and stored results. Keep those inputs immutable. It can rebuild a
missing or damaged partial journal from committed checkpoints. An existing final
receipt is a completed evaluation, not a target for resume.

``--workers N`` supports 1–8 isolated workers and preserves ordered results. Worker
count may change during resume, but concurrency uses additional memory and may not
improve throughput. Start with one worker and measure your workload.

Keep the report usable
----------------------

The final receipt contains aggregate cells and an identity for its detailed-result
journal. Keep both files together, preserving the recorded relative journal path.
The SQLite ledger is needed for resume, not for reading a completed report.
``modebench.api.read_report`` and ``iter_results`` verify the journal hash.

Allocate storage for the raw responses, journal, and checkpoints. Durability relies
on filesystem locking and ``fsync``. The verifier processes have bounded deadlines
and memory; this is not a general-purpose sandbox for arbitrary programs.
