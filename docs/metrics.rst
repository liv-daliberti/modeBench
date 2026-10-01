Understand the scores
=====================

For one prompt with ``k`` responses, let ``K`` be the number that verify and
``n_m`` the number of verified responses in mode ``m``.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Per-prompt value
     - Aggregation
   * - Accuracy
     - ``K / k``
     - Equal mean over prompts.
   * - Empirical pass@k
     - One if any response verifies; otherwise zero.
     - Equal mean over prompts.
   * - Distinct@k
     - Number of different verified mode identities.
     - Equal mean over prompts.
   * - PCMD
     - ``1 - sum(n_m * (n_m - 1)) / (K * (K - 1))``
     - Equal mean over prompts with at least two successes.

PCMD is **pairwise correct-mode diversity**: the probability that two distinct
verified draws have different mode identities. Two different verified modes give
one; repeated successes in a single mode give zero. Counts ``{A: 2, B: 1}`` give
``2/3``. Fewer than two successes gives an undefined value, not zero diversity.

Reporting support
-----------------

Correctness affects which prompts contribute to PCMD. Always report accuracy,
the eligible-prompt count, total prompt count, and the ``reportable`` flag beside
it. The default reporting threshold is 30 eligible prompts. A numeric value can
exist with ``reportable=false``; do not present that as a supported benchmark score.

Changing ``--min-defined-prompts`` changes the reporting rule and must be declared
before analysis. ``standard_error`` is the sample standard deviation of eligible
prompt scores divided by the square root of their count, not a paired treatment
effect interval. It is undefined for one eligible prompt.

Repeated groups
---------------

The saved-response CLI treats a prompt's response list as one group. The metric API
also accepts repeated groups:

.. doctest::

   >>> from modebench.metrics import POOLED, PER_GROUP, prompt_mode_diversity
   >>> draws = [{"attempts": [
   ...     {"verified": True, "canonical_key": "A"},
   ...     {"verified": True, "canonical_key": "B"},
   ...     {"verified": False, "canonical_key": None},
   ... ]}]
   >>> prompt_mode_diversity(draws, POOLED)
   1.0
   >>> prompt_mode_diversity(draws, PER_GROUP)
   1.0

``POOLED`` pools a prompt's groups; ``PER_GROUP`` averages its defined group scores.
Never pool canonical identities across prompts. Four groups of eight responses
can support pass@8 averaged over groups and PCMD pooled within each prompt; flattening
them for correctness instead measures pass@32.

``rarefied_distinct_at_2`` equals ``1 + PCMD``. ``effective_modes`` is
``1 / (1 - mean_PCMD)`` after aggregation; it is undefined when PCMD is undefined
or equals one and is not an unbiased estimate of the number of modes.
