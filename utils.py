from random import random


def get_random(min_limit: float = 0.0, max_limit: float = 1.0) -> float:
    """Return a random float uniformly distributed in [min_limit, max_limit]."""
    return min_limit + random() * (max_limit - min_limit)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to the closed interval [min_val, max_val]."""
    return max(min_val, min(max_val, value))
