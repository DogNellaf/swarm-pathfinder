from __future__ import annotations

import random
from typing import Callable, List, Tuple

from settings import ALPHA_MAX, ALPHA_MIN, BETA_MAX, BETA_MIN, LANDSCAPES_COUNT
from utils import clamp, get_random


class Particle:
    """A single PSO particle representing a candidate (alpha, beta) solution.

    Position layout: [alpha_0, alpha_1, ..., alpha_{n-1}, beta]
    Velocity has the same dimensionality as position.
    """

    def __init__(
        self,
        eval_fn: Callable[[List[float], float], float],
        num_alpha: int = LANDSCAPES_COUNT,
        alpha_bounds: Tuple[float, float] = (ALPHA_MIN, ALPHA_MAX),
        beta_bounds: Tuple[float, float] = (BETA_MIN, BETA_MAX),
    ) -> None:
        self.eval_fn = eval_fn
        self.num_alpha = num_alpha
        self.total_dim = num_alpha + 1  # alpha dims + beta

        # [alpha_0..alpha_{n-1}, beta]
        self.position: List[float] = [
            get_random(*alpha_bounds) for _ in range(num_alpha)
        ] + [get_random(*beta_bounds)]

        self.velocity: List[float] = [0.0] * self.total_dim

        # per-dimension (lo, hi) used for position clamping
        self._bounds: List[Tuple[float, float]] = (
            [alpha_bounds] * num_alpha + [beta_bounds]
        )

        self.best_w: float = float("inf")
        # Initialise best_position before first evaluate so it is always set
        self.best_position: List[float] = self.position.copy()

        self.current_w: float = self.evaluate()

    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        alpha = self.position[:-1]
        beta = self.position[-1]
        self.current_w = self.eval_fn(alpha, beta)
        if self.current_w < self.best_w:
            self.best_w = self.current_w
            # Copy so that subsequent position mutations do not corrupt history
            self.best_position = self.position.copy()
        return self.current_w

    def update_velocity(
        self,
        informant_best_position: List[float],
        inertia: float,
        cognitive: float,
        social: float,
    ) -> None:
        """Standard PSO velocity update followed by position clamping.

        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        """
        r1 = random.random()
        r2 = random.random()

        for i in range(self.total_dim):
            self.velocity[i] = (
                inertia * self.velocity[i]
                + cognitive * r1 * (self.best_position[i] - self.position[i])
                + social * r2 * (informant_best_position[i] - self.position[i])
            )

        for i in range(self.total_dim):
            lo, hi = self._bounds[i]
            self.position[i] = clamp(self.position[i] + self.velocity[i], lo, hi)
