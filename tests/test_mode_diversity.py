"""Contract for the success-conditional modal diversity metric."""
from math import comb, isclose
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))

from modebench.metrics import (  # noqa: E402
    DEFAULT_MIN_DEFINED_PROMPTS,
    PER_GROUP,
    POOLED,
    cell_summary,
    effective_modes,
    mode_diversity,
    prompt_mode_diversity,
    rarefied_distinct,
    verified_mode_counts,
)


def _draw(*keys):
    """A group of eight attempts; ``None`` marks an unverified sample."""
    attempts = [{"canonical_key": k, "verified": k is not None} for k in keys]
    attempts += [{"canonical_key": None, "verified": False}] * (8 - len(keys))
    return {"attempts": attempts}


@pytest.mark.parametrize("counts", [[4, 2, 1, 1], [8], [1, 1], [3, 3], [2, 1, 1, 1], [5, 3]])
def test_matches_the_pair_counting_definition(counts):
    """D_mode = 1 - sum_m C(n_m,2) / C(K,2), computed directly from the definition."""
    total = sum(counts)
    expected = 1 - sum(comb(n, 2) for n in counts) / comb(total, 2)
    assert isclose(mode_diversity(counts), expected, rel_tol=1e-12)


def test_is_the_fraction_of_cross_mode_pairs():
    counts = [4, 2, 1, 1]
    labels = [m for m, n in enumerate(counts) for _ in range(n)]
    pairs = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    different = sum(labels[i] != labels[j] for i, j in pairs)
    assert isclose(mode_diversity(counts), different / len(pairs), rel_tol=1e-12)


def test_undefined_below_two_verified_samples():
    """Zero or one success carries no pair, so the metric must abstain."""
    assert mode_diversity([]) is None
    assert mode_diversity([1]) is None
    assert mode_diversity([0, 0]) is None
    assert mode_diversity({"a": 1}) is None


def test_bounds_are_collapse_and_full_spread():
    assert mode_diversity([6]) == 0.0          # every success the same mode
    assert mode_diversity([1, 1, 1, 1]) == 1.0  # every success a different mode


def test_is_independent_of_accuracy():
    """Scaling the number of failures cannot move the metric."""
    verified = ["a", "a", "b"]
    sparse = prompt_mode_diversity([_draw(*verified)], POOLED)
    dense = prompt_mode_diversity([_draw(*verified), _draw()], POOLED)
    assert isclose(sparse, dense, rel_tol=1e-12)


def test_unbiased_for_gini_simpson_under_iid_sampling():
    """Averaging over all outcomes of K i.i.d. draws recovers 1 - sum q^2 exactly."""
    from itertools import product
    from collections import Counter

    q = {"a": 0.5, "b": 0.3, "c": 0.2}
    target = 1 - sum(v * v for v in q.values())
    for draws in (2, 3, 4):
        total = 0.0
        for outcome in product(q, repeat=draws):
            weight = 1.0
            for key in outcome:
                weight *= q[key]
            total += weight * mode_diversity(Counter(outcome))
        assert isclose(total, target, rel_tol=1e-12)


@pytest.mark.parametrize("counts", [[4, 2, 1, 1], [3, 3], [1, 1], [5, 3], [2, 2, 2]])
def test_rarefied_distinct_at_two_is_one_plus_diversity(counts):
    """Pairwise diversity is a success-conditional distinct@2."""
    assert isclose(rarefied_distinct(counts, 2), 1 + mode_diversity(counts), rel_tol=1e-12)


def test_rarefaction_matches_its_closed_form_and_needs_depth_many_successes():
    counts = [4, 2, 1, 1]
    total = sum(counts)
    expected = sum(1 - comb(total - n, 3) / comb(total, 3) for n in counts)
    assert isclose(rarefied_distinct(counts, 3), expected, rel_tol=1e-12)
    assert rarefied_distinct([1, 1], 3) is None
    assert rarefied_distinct([9], 3) == 1.0
    with pytest.raises(ValueError):
        rarefied_distinct(counts, 0)


def test_rarefaction_is_nondecreasing_in_depth():
    counts = [4, 2, 1, 1]
    values = [rarefied_distinct(counts, r) for r in (1, 2, 3, 4)]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))


def test_effective_modes_inverts_diversity_and_abstains_when_it_diverges():
    assert isclose(effective_modes(0.0), 1.0, rel_tol=1e-12)
    assert isclose(effective_modes(0.5), 2.0, rel_tol=1e-12)
    assert effective_modes(1.0) is None  # every success distinct; 1/(1-D) unbounded
    assert effective_modes(None) is None


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError):
        mode_diversity([2, -1])
    with pytest.raises(ValueError):
        rarefied_distinct([2, -1], 2)


def test_verified_mode_counts_ignores_failures():
    counts = verified_mode_counts(_draw("a", "a", "b")["attempts"])
    assert dict(counts) == {"a": 2, "b": 1}


def test_pooling_and_per_group_aggregation_differ_as_documented():
    """A prompt with one success per group is defined pooled, undefined per group."""
    draws = [_draw("a"), _draw("b"), _draw("c"), _draw("d")]
    assert prompt_mode_diversity(draws, PER_GROUP) is None
    assert isclose(prompt_mode_diversity(draws, POOLED), 1.0, rel_tol=1e-12)
    with pytest.raises(ValueError):
        prompt_mode_diversity(draws, "nonsense")


def test_cell_summary_reports_support_and_gates_on_it():
    defined = {"draws": [_draw("a", "b")]}
    undefined = {"draws": [_draw("a")]}
    summary = cell_summary([defined] * 4 + [undefined] * 6, POOLED, min_defined_prompts=4)
    assert summary["defined_prompts"] == 4
    assert summary["prompts"] == 10
    assert isclose(summary["support"], 0.4, rel_tol=1e-12)
    assert summary["reportable"] is True
    assert isclose(summary["d_mode"], 1.0, rel_tol=1e-12)
    assert isclose(summary["rarefied_distinct_at_2"], 2.0, rel_tol=1e-12)
    assert summary["effective_modes"] is None  # D_mode == 1 here

    gated = cell_summary([defined] * 4 + [undefined] * 6, POOLED, min_defined_prompts=5)
    assert gated["reportable"] is False, "cells below the support bar are gaps, not numbers"


def test_cell_summary_handles_a_cell_with_no_verified_pairs():
    summary = cell_summary([{"draws": [_draw()]}] * 5, POOLED)
    assert summary["d_mode"] is None and summary["defined_prompts"] == 0
    assert summary["reportable"] is False
    assert summary["support"] == 0.0


def test_agrees_with_the_registered_collision_estimator():
    """PMD is 1 - collision, so the two implementations must never drift.

    ``analyze_paper_conditional_concentration.collision`` is the registered
    conditional-concentration U-statistic already used for the training
    blocks. This metric is its complement, and the manuscript now reports the
    diversity orientation, so an inconsistency between them would put two
    contradictory numbers in the same paper.
    """
    import random
    from collections import Counter

    if str(ROOT / "ops" / "exp_scaling") not in sys.path:
        sys.path.insert(0, str(ROOT / "ops" / "exp_scaling"))
    from collision_reference import collision

    random.seed(20260914)
    for _ in range(500):
        counts = Counter()
        for _ in range(random.randint(0, 12)):
            counts[random.choice("abcde")] += 1
        diversity, concentration = mode_diversity(counts), collision(counts)
        if diversity is None or concentration is None:
            assert diversity is None and concentration is None
        else:
            assert isclose(diversity, 1 - concentration, abs_tol=1e-12)


def test_default_support_bar_is_declared():
    assert DEFAULT_MIN_DEFINED_PROMPTS == 30
