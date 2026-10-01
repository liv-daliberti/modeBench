Discover tasks and collect responses
====================================

Find and inspect a split
------------------------

.. code-block:: console

   python -m pip install 'modebench[data]==0.4.0'
   modebench datasets list
   modebench datasets card --config level1_countdown
   modebench datasets fetch --config level1_countdown
   modebench datasets inspect --config level1_countdown --row 0 --offline

The registry contains **72 frozen splits / 15,552 rows**. The first Countdown
evaluation task has ID ``level1_countdown/eval/0``; its split has 128 rows.
A dataset card explains task semantics, mode identity, fields, authorship, and licenses.

``fetch`` verifies SHA-256 identities shipped with the package. The default cache is
``$XDG_CACHE_HOME/modebench`` or ``~/.cache/modebench``, separated by registry identity.
Every reuse checks the bytes. Use ``--cache-dir PATH`` for writable scratch storage
and ``--offline`` to forbid downloads. A corrupt cache entry is an error; remove
that named file and fetch the same split again.

Domains and solution modes
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Domain
     - Correctness
     - Canonical identity
   * - Graph coloring
     - Edges and fixed colors are respected.
     - The full color vector.
   * - Countdown
     - Exact arithmetic uses the given operands.
     - A normalized executed expression tree.
   * - Python factors
     - A restricted function returns proper divisors on the reference inputs.
     - The returned divisor vector.
   * - MathIR
     - Restricted equation actions execute correctly.
     - The exact normalized state trajectory.
   * - PantryPlan
     - Ingredient and nutrition constraints hold.
     - Ingredient support.

Mode identity belongs to a single prompt. Different wording does not create a
new mode. Level-1 Pantry accepts a six-bit support mask and projects it to an
allocation deterministically; later Pantry levels use allocation responses.

Levels and split sizes
----------------------

* Level 1 is the native task suite.
* Level 2 targets a harder regime for small models.
* Levels 3, 4, and 5 calibrate task mixtures at Qwen2.5 3B, 7B, and 14B respectively.
  Their construction matches recorded correctness targets, not diversity targets.

Level-1 domains have 384 training and 128 evaluation rows. Pantry also has a
64-row development split. Levels 2–5 have 384/128/128 train/dev/eval rows per domain.
The separate Level-1 graph single-answer diagnostic is not part of the main suite.

.. important::
   **Level 4 is not admitted overall:** MathIR failed difficulty matching. The other
   four domains passed. Levels change task populations and sometimes prompt guidance;
   their numbers are not a pure causal comparison of difficulty. See :doc:`results`
   for the detailed construction and calibration records.

Prepare requests without leaking answers
----------------------------------------

.. code-block:: console

   modebench prepare --config level1_countdown --output requests.jsonl

Each line carries a stable ID, registered chat messages, and prompt/dataset hashes.
Send only its ``messages`` to your model. Preserve the exported record and add a
``responses`` list containing the raw model outputs in sampling order. Use the same
number of responses for every prompt, including wrong answers and refusals.

The ``Task`` API object contains the grading reference: never send that object,
``answer``, certified solutions, or support counts to the model. Inspection hides
references unless ``--show-reference`` is requested explicitly.

Create run metadata
-------------------

Save a JSON object as ``run.json``, replacing every illustrative identity with the
actual model revision and generation settings:

.. code-block:: json

   {
     "model": {"id": "your-model", "revision": "exact-model-revision"},
     "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 192, "seed": 43},
     "prompt_condition": "registered_hints_v1"
   }

Metadata declares how responses were collected; it is not proof of the actual model
transport. The registered messages do not reproduce historical decoding masks.
Record whether decoding was free or syntax-constrained.

Continue with :doc:`evaluation` to check the responses against the frozen split.
