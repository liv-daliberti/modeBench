"""Registered collision estimator, extracted verbatim from the research analysis."""
def collision(counts):
    """Unbiased collision U-statistic for fixed-policy iid correct labels."""
    values = list(counts.values())
    if any(type(n) is not int or n < 0 for n in values):
        raise ValueError('key counts must be nonnegative integers')
    n = sum(values)
    return sum(v * (v - 1) for v in values) / (n * (n - 1)) if n >= 2 else None
