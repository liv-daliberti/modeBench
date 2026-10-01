Install and try all five domains
================================

Install
-------

Use Linux and CPython 3.10–3.12. WSL provides a Linux environment on Windows;
native Windows and macOS are not qualified. Choose a new working directory:

.. code-block:: console

   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install modebench==0.4.0

The core package supplies all five verifiers, metrics, the CLI, dataset discovery,
and the walkthrough. Reading real Parquet datasets additionally needs
``python -m pip install 'modebench[data]==0.4.0'``.

Create saved responses
----------------------

.. code-block:: console

   modebench walkthrough --directory demo
   head -n 1 demo/tasks.jsonl
   head -n 1 demo/responses.jsonl

The directory contains one task per domain, four handwritten responses per task,
run metadata, and an expected-results file. Each task has two correct modes, one
incorrect response, and one malformed response. No model calls or downloads occur.

``tasks.jsonl`` lets you inspect the problems. ``responses.jsonl`` is the evaluator
input; ``run.json`` records the example's model/generation condition. To try your
own answers, copy ``responses.jsonl`` and replace the strings in its ``responses``
lists. Preserve wrong and malformed attempts rather than removing them.

Evaluate and read the report
----------------------------

.. code-block:: console

   modebench evaluate demo/responses.jsonl --run demo/run.json --output demo/report.json
   modebench report demo/report.json

The report contains five rows with these values:

.. code-block:: text

   level domain          k accuracy pass@k distinct@k PCMD reportable
   1     countdown       4 0.5      1      2          1    false
   1     graph_coloring  4 0.5      1      2          1    false
   1     mathir          4 0.5      1      2          1    false
   1     pantry_plan     4 0.5      1      2          1    false
   1     python_factors  4 0.5      1      2          1    false

Accuracy is two correct responses out of four. Each task is solved at least once,
and its two verified responses have different canonical identities. The example
has only one eligible prompt per domain, so ``reportable=false`` is expected:
the default reporting threshold is 30 eligible prompts.

.. important::
   This is a custom-reference interface demonstration, not a full benchmark result.
   Keep the final receipt and its detailed-result journal together. Existing output
   files are never overwritten; use a fresh filename for a changed set of responses.

Next, :doc:`datasets` shows how to fetch a real frozen split and prepare model requests.
