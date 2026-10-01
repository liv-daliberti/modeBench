Performance and reproducibility
===============================

Untrained-model performance
---------------------------

Mean **pass@8** over all five domains for Qwen2.5-Instruct before RL training.
Each cell uses 128 held-out prompts per domain and four independent groups of
eight responses. This table is read directly from the evidence-checked README.

.. readme-table:: modebench-performance

The dagger marks Level 4, which includes the unmatched MathIR condition and is
not admitted overall. The grid uses native chat, temperature 1, top-p 1, a 192-token
limit, boxed-direct Graph prompts, and guided decoding in the other domains.
Changing those conditions changes the comparison.

Recompute the retained results
------------------------------

From a checkout:

.. code-block:: console

   git clone https://github.com/liv-daliberti/modeBench.git
   cd modeBench
   python ops/reproduce_base_grid.py
   python ops/summarize_levels.py --check

The first command recomputes 375 cells from retained correctness flags and
canonical keys. It does not regenerate model outputs or independently regrade
raw generations. The second checks the displayed calibration, base-grid, and
training-summary tables against retained evidence.

The `detailed level tables <https://github.com/liv-daliberti/modeBench#levels-and-results>`_
contain tuning outcomes, all 25 Qwen-7B level/domain cells, and recorded training
endpoints for Levels 1–3. The
`construction summary <https://github.com/liv-daliberti/modeBench/blob/main/provenance/level-construction.json>`_
records population boundaries, mixture fitting, and source identities.

Training comparisons
--------------------

Re:Dr and Re:Max training live in the separate ReMax package. Its
`comparison guide <https://liv-daliberti.github.io/remax/results.html>`_
shows the maintained methods, GRPO, other model scales, and retained ablations.
It distinguishes runnable training from saved-key reproduction and summary-only
snapshots. Untrained-model and training results require matching evaluation
protocols before interpreting a difference as a training effect.

What to record with a result
----------------------------

Keep the software version, verifier contract, dataset configuration/split/hash,
prompt condition, model revision, generation settings, seeds, draw budget,
coverage, exclusions, receipt, and detailed journal. Frozen-data identity does
not by itself establish that the actual model request used the registered messages.

Changes to accepted responses, canonical identities, prompts, split contents,
aggregation, or failure policy require a scientific compatibility decision and
new identities where their meaning changes. Do not repair a failing reference
comparison by overwriting its expected values.
