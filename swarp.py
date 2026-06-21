"""Backward-compatible wrapper around Swarm.

The original class was called Swarp (typo for Swarm) and exposed .a and .b
attributes holding the first alpha coefficient and beta of the best solution
found.  This shim preserves that interface so existing notebooks continue to
work without modification.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from swarm import Swarm
from w_solution import get_W


class Swarp(Swarm):
    """Legacy alias for Swarm with .a/.b attributes for notebook compatibility."""

    def __init__(
        self,
        np: int,
        ni_min: int,
        ni_max: int,
        k: int,
        weight: float,
        a: float,
        b: float,
    ) -> None:
        # a and b here are PSO cognitive/social coefficients (not alpha/beta)
        super().__init__(
            eval_fn=get_W,
            swarm_size=np,
            neighbor_min=ni_min,
            neighbor_max=ni_max,
            max_iterations=k,
            inertia=weight,
            cognitive=a,
            social=b,
        )
        # Will be populated by optimize()
        self.a: Optional[float] = None
        self.b: Optional[float] = None

    def optimize(self) -> float:
        best_w = super().optimize()
        alpha, beta, _ = self.get_best_solution()
        # Expose the best first-alpha and beta for notebook compatibility
        self.a = alpha[0]
        self.b = beta
        return best_w
