"""Small statistics helpers shared across environment modules."""

from typing import Iterable


def gini(values: Iterable[float]) -> float:
    """
    Gini coefficient of non-negative values.

    0 = perfect equality, 1 = maximum inequality. Returns 0.0 for empty
    input or when all values are zero.
    """
    vals = sorted(values)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    gini_sum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(vals))
    return gini_sum / (n * total)
