from __future__ import annotations

import math
from random import choices, randint
from typing import Callable, List, Optional, Tuple

from particle import Particle
from settings import (
    ALPHA_MAX,
    ALPHA_MIN,
    BETA_MAX,
    BETA_MIN,
    CURRENT_VELOCITY_RATIO,
    GLOBAL_VELOCITY_RATIO,
    LANDSCAPES_COUNT,
    LOCAL_VELOCITY_RATIO,
    STEPS_COUNT,
    SWARM_SIZE,
)
from utils import get_random


class Swarm:
    """Particle Swarm Optimiser that minimises an arbitrary scalar objective.

    Each particle represents a candidate (alpha, beta) vector.  In every
    iteration each particle selects a random neighbourhood, finds the
    informant with the lowest personal-best W, and updates its velocity
    toward that informant's best-known position.  After the velocity update
    the particle evaluates its new position so that learning actually occurs.
    """

    def __init__(
        self,
        eval_fn: Callable[[List[float], float], float],
        swarm_size: int = SWARM_SIZE,
        neighbor_min: int = 15,
        neighbor_max: int = 25,
        max_iterations: int = STEPS_COUNT,
        inertia: float = CURRENT_VELOCITY_RATIO,
        cognitive: float = LOCAL_VELOCITY_RATIO,
        social: float = GLOBAL_VELOCITY_RATIO,
        num_alpha: int = LANDSCAPES_COUNT,
        alpha_bounds: Tuple[float, float] = (ALPHA_MIN, ALPHA_MAX),
        beta_bounds: Tuple[float, float] = (BETA_MIN, BETA_MAX),
    ) -> None:
        self.neighbor_min = neighbor_min
        self.neighbor_max = neighbor_max
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.particles: List[Particle] = [
            Particle(eval_fn, num_alpha, alpha_bounds, beta_bounds)
            for _ in range(swarm_size)
        ]

        self.best_w: float = float("inf")
        self.best_position: Optional[List[float]] = None
        self._convergence_history: List[float] = []

        self._update_global_best()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_global_best(self) -> None:
        for p in self.particles:
            if p.best_w < self.best_w:
                self.best_w = p.best_w
                self.best_position = p.best_position.copy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self) -> float:
        """Run the PSO loop and return the best W found."""
        self._convergence_history = []

        for _ in range(self.max_iterations):
            for particle in self.particles:
                ni = randint(self.neighbor_min, self.neighbor_max)
                neighbors = choices(self.particles, k=ni)

                # Best informant: neighbour with lowest personal-best W
                best_informant = min(neighbors, key=lambda n: n.best_w)

                particle.update_velocity(
                    best_informant.best_position,
                    self.inertia,
                    self.cognitive,
                    self.social,
                )
                particle.evaluate()

            self._update_global_best()
            self._convergence_history.append(self.best_w)

        return self.best_w

    def get_best_solution(self) -> Tuple[List[float], float, float]:
        """Return (alpha_vector, beta, best_w) for the best solution found."""
        if self.best_position is None:
            raise RuntimeError("Call optimize() before get_best_solution().")
        alpha = self.best_position[:-1]
        beta = self.best_position[-1]
        return alpha, beta, self.best_w

    def get_standard_deviation(self) -> float:
        """Population standard deviation of personal-best W values."""
        w_values = [p.best_w for p in self.particles]
        n = len(w_values)
        mean = sum(w_values) / n
        variance = sum((w - mean) ** 2 for w in w_values) / n
        return math.sqrt(variance)

    def get_convergence_history(self) -> List[float]:
        """Global-best W at the end of each iteration (after optimize())."""
        return self._convergence_history.copy()


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from w_solution import PathFinder
    from settings import LANDSCAPES_PATH, MAP_PATH, PARAMS_PATH

    finder = PathFinder.from_files(LANDSCAPES_PATH, MAP_PATH, PARAMS_PATH)

    swarm = Swarm(
        eval_fn=finder.get_W,
        swarm_size=30,
        neighbor_min=15,
        neighbor_max=25,
        max_iterations=200,
        inertia=0.01,
        cognitive=get_random(0.5, 0.95),
        social=get_random(0.5, 0.95),
    )

    best_w = swarm.optimize()
    alpha, beta, _ = swarm.get_best_solution()

    print(f"Best W       = {best_w:.4f}")
    print(f"Std deviation = {swarm.get_standard_deviation():.4f}")
    print(f"Alpha        = {[round(a, 4) for a in alpha]}")
    print(f"Beta         = {beta:.4f}")
