#!/usr/bin/env python3
"""Conditional pairwise correct-mode diversity: a breadth metric independent of accuracy.

``distinct@8`` conflates two things. A model that never succeeds scores zero
breadth because it never succeeds, not because its successes are concentrated.
Across the 60 frozen base-model cells ``distinct@8`` correlates .967 with
``pass@8``: as reported it is very nearly a restatement of accuracy.

This module implements the success-conditional alternative. For one prompt, let
``K`` of the generations be verified and let ``n_1..n_M`` be the counts of those
verified generations falling in each canonical mode, so ``sum_m n_m = K``. Then

    D_mode = 1 - sum_m C(n_m, 2) / C(K, 2)                          (K >= 2)

which is the fraction of *pairs* of verified samples that occupy different
modes, i.e. the probability that two independently drawn verified solutions to
the same prompt are different modes. Under i.i.d. sampling it is the unbiased
estimator of the Gini-Simpson diversity ``1 - sum_m q_m^2`` of the
success-conditional mode distribution ``q``, where the total success
probability ``p`` has cancelled out. It is therefore a functional of ``q``
alone and carries no accuracy signal by construction.

Two consequences follow and are handled explicitly below.

``D_mode`` is undefined at ``K < 2``: one verified sample has no pair, and zero
verified samples reveal nothing about how successes would be distributed. This
is an information limit, not a defect -- a model that almost never succeeds has
no measurable success diversity. Because definedness itself correlates .99 with
``pass@8``, every reported value must carry its support, and cells below a
stated support bar are reported as explicit gaps rather than as noise. See
``cell_summary``.

The same statistic has two other faces, both provided here. Fixed-depth
rarefaction asks how many distinct modes would be seen in exactly ``r``
verified draws; at ``r = 2`` that expectation is ``1 + D_mode``, so pairwise
diversity is literally ``distinct@2`` after conditioning on success and fixing
the sampling depth. This answers the objection that different models reach
different ``K``. And ``N_eff = 1 / (1 - D_mode)`` is the effective number of
verified modes, the count-valued reading of the same quantity. ``N_eff``
diverges as ``D_mode`` approaches one, so it is derived from an already
aggregated ``D_mode`` rather than averaged over prompts; see ``effective_modes``.

``q`` is a property of the policy and prompt, not of the draw budget, so the
estimator does not have to be confined to a single group of eight. Pooling a
prompt's independent groups estimates the same quantity with more pairs and
much better definedness; ``POOLED`` does this and ``PER_GROUP`` follows the
eight-draw group convention used elsewhere in the paper.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence
import statistics
import math
from numbers import Integral, Real
from .validation import positive_integer
from .diagnostics import STATUSES, UNSCORABLE, VerifierExecutionError

PER_GROUP = 'per_group'
POOLED = 'pooled'
AGGREGATIONS = (PER_GROUP, POOLED)

#: Cells with fewer defined prompts than this are reported as gaps, not numbers.
DEFAULT_MIN_DEFINED_PROMPTS = 30


def _counts(counts):
    values = list(counts.values()) if isinstance(counts, Mapping) else list(counts)
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in values):
        raise ValueError('verified mode counts must be nonnegative integers')
    return [int(n) for n in values]


def mode_diversity(counts: Mapping[Any, int] | Iterable[int]) -> float | None:
    """Return ``D_mode`` for one prompt's verified-mode counts, or ``None``.

    ``counts`` gives ``n_1..n_M``, the number of verified samples in each
    canonical mode; it may be a mapping keyed by mode or a bare iterable of
    counts. Returns ``None`` when fewer than two verified samples are present,
    which is the only case where the quantity is undefined.
    """
    values = _counts(counts)
    total = sum(values)
    if total < 2:
        return None
    concentration = sum(n * (n - 1) for n in values) / (total * (total - 1))
    return 1.0 - concentration


def rarefied_distinct(counts: Mapping[Any, int] | Iterable[int], depth: int = 2) -> float | None:
    """Expected distinct modes among exactly ``depth`` verified draws, or ``None``.

    This is the unbiased fixed-depth rarefaction estimator
    ``sum_m [1 - C(K - n_m, r) / C(K, r)]``, defined when ``K >= r``. At
    ``depth == 2`` it equals ``1 + mode_diversity(counts)`` identically, which
    is the sense in which pairwise correct-mode diversity is a success-conditional
    ``distinct@2``. Larger depths resolve more structure but are defined on
    fewer prompts, so they serve as robustness checks rather than headline
    numbers.
    """
    depth = positive_integer(depth, 'rarefaction depth')
    values = _counts(counts)
    total = sum(values)
    if total < depth:
        return None
    expected = 0.0
    for n in values:
        if n == 0:
            continue
        # Probability that mode ``n`` is missed by all ``depth`` draws.
        missed = 1.0
        for i in range(depth):
            missed *= (total - n - i) / (total - i)
        expected += 1.0 - missed
    return expected


def effective_modes(d_mode: float | None) -> float | None:
    """Effective number of verified modes ``N_eff = 1 / (1 - D_mode)``.

    Pass an already aggregated ``D_mode``. Applying this per prompt and then
    averaging would be dominated by prompts whose few verified draws happened
    to be all distinct, where ``D_mode == 1`` sends ``N_eff`` to infinity;
    returns ``None`` in that degenerate case rather than a sentinel.
    """
    if d_mode is None:
        return None
    if isinstance(d_mode, bool) or not isinstance(d_mode, Real) or not math.isfinite(d_mode) or not 0 <= d_mode <= 1:
        raise ValueError('d_mode must be finite and between zero and one')
    if d_mode == 1.0:
        return None
    return 1.0 / (1.0 - d_mode)


def verified_mode_counts(attempts: Sequence[Mapping[str, Any]]) -> Counter:
    """Count verified attempts by canonical key, ignoring failures."""
    counts = Counter()
    for attempt in attempts:
        if isinstance(attempt, Mapping) and attempt.get('status') in UNSCORABLE:
            raise VerifierExecutionError(dict(attempt))
        if not isinstance(attempt, Mapping) or type(attempt.get('verified')) is not bool:
            raise ValueError('each attempt requires a boolean verified flag')
        status = attempt.get('status')
        if status is not None and (status not in STATUSES or attempt['verified'] != (status == 'correct')):
            raise ValueError('attempt status and verified flag disagree')
        if attempt['verified']:
            key = attempt.get('canonical_key')
            if not isinstance(key, str) or not key:
                raise ValueError('verified attempts require a nonempty canonical_key string')
            counts[key] += 1
    return counts


def prompt_mode_diversity(draws: Sequence[Mapping[str, Any]],
                          aggregation: str = POOLED) -> float | None:
    """Return one prompt's ``D_mode``, or ``None`` when it is undefined.

    Under ``POOLED`` the prompt's groups are pooled into a single verified-mode
    count vector. Under ``PER_GROUP`` each group is scored separately and the
    defined groups are averaged, matching the surrounding convention of
    averaging groups within a prompt before averaging prompts.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f'unknown aggregation: {aggregation!r}')
    if aggregation == POOLED:
        pooled: Counter = Counter()
        for draw in draws:
            pooled.update(verified_mode_counts(draw['attempts']))
        return mode_diversity(pooled)
    scored = [mode_diversity(verified_mode_counts(draw['attempts'])) for draw in draws]
    defined = [value for value in scored if value is not None]
    return statistics.fmean(defined) if defined else None


def cell_summary(prompt_results: Sequence[Mapping[str, Any]],
                 aggregation: str = POOLED,
                 min_defined_prompts: int = DEFAULT_MIN_DEFINED_PROMPTS) -> dict:
    """Summarise one model--domain--level cell with its support disclosed.

    ``d_mode`` is the unweighted mean over prompts where the metric is defined.
    Prompts are weighted equally regardless of how many verified samples they
    contributed, because weighting by verified count would reintroduce the
    accuracy dependence this metric exists to remove.

    ``reportable`` is False when the cell has too few defined prompts to carry
    a number; such cells are rendered as explicit gaps. The support fields are
    part of the result, not diagnostics, and are meant to travel with the value
    wherever it is printed.
    """
    min_defined_prompts = positive_integer(min_defined_prompts, 'min_defined_prompts')
    if aggregation not in AGGREGATIONS:
        raise ValueError(f'unknown aggregation: {aggregation!r}')
    values = [prompt_mode_diversity(result['draws'], aggregation)
              for result in prompt_results]
    defined = [value for value in values if value is not None]
    prompts = len(values)
    return {
        'aggregation': aggregation,
        'd_mode': statistics.fmean(defined) if defined else None,
        'defined_prompts': len(defined),
        'prompts': prompts,
        'support': len(defined) / prompts if prompts else 0.0,
        'min_defined_prompts': min_defined_prompts,
        'reportable': len(defined) >= min_defined_prompts,
        'standard_error': (statistics.stdev(defined) / len(defined) ** 0.5
                           if len(defined) > 1 else None),
        'rarefied_distinct_at_2': (1.0 + statistics.fmean(defined)) if defined else None,
        'effective_modes': effective_modes(statistics.fmean(defined) if defined else None),
    }
