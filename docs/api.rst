Python API
==========

Use ``modebench.api`` for new integrations. These references are generated from
the installed 0.4.0 package, without importing the research checkout.

Grade a task
------------

.. doctest::

   >>> from modebench.api import Task, grade
   >>> task = Task(
   ...     id="example", level=1, domain="countdown",
   ...     problem="Use 1, 2, 3 exactly once to make 6.",
   ...     answer={"verifier": "countdown", "numbers": [1, 2, 3], "target": 6},
   ... )
   >>> result = grade(task, "1+2+3")
   >>> result["status"], result["verified"]
   ('correct', True)
   >>> grade(task, "1+2-3")["status"]
   'incorrect'

This custom task illustrates the interface. For a frozen task, install the data
extra and load authenticated rows:

.. code-block:: python

   from modebench.api import load_tasks, make_prompt, grade

   task = next(load_tasks("level1_countdown"))
   messages = make_prompt(task)
   # Send only messages to your model; keep task.answer in the evaluator.
   result = grade(task, "YOUR_MODEL_RESPONSE")
   print(result["status"], result["canonical_key"], result["detail"])

``grade`` returns structured diagnostics. Inspect the status before using a reward
or mode identity; evaluator failures must not be treated as incorrect responses.

.. autofunction:: modebench.api.load_tasks

.. autofunction:: modebench.api.make_prompt

.. autofunction:: modebench.api.grade

.. autoclass:: modebench.api.Task

Evaluate and inspect a response file
------------------------------------

.. code-block:: python

   from modebench.api import EvaluationOptions, evaluate_file, read_report, iter_results

   receipt = evaluate_file(
       "responses.jsonl", "results.json",
       options=EvaluationOptions(min_defined_prompts=30),
   )
   report = read_report(receipt.output)
   if report.status != "completed":
       raise RuntimeError("Evaluation failed; inspect the detailed diagnostics")
   for cell in report.cells:
       print(cell.domain, cell.accuracy, cell.pcmd.reportable)
   for prompt in iter_results(receipt.output):
       print(prompt["id"], prompt["accuracy"])

.. autofunction:: modebench.api.evaluate_file

.. autofunction:: modebench.api.read_report

.. autofunction:: modebench.api.iter_results

.. autoclass:: modebench.api.EvaluationOptions

.. autoclass:: modebench.api.EvaluationReceipt

.. autoclass:: modebench.api.Report

The package ships ``py.typed``. Data classes and typed dictionaries describe task,
grade, receipt, report, and progress fields; legacy private helpers are outside the
supported public contract.
