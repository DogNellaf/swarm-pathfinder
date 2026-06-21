"""Entry point: load data, run PSO, and print results."""
from __future__ import annotations

from settings import (
    CURRENT_VELOCITY_RATIO,
    GLOBAL_VELOCITY_RATIO,
    LANDSCAPES_PATH,
    LOCAL_VELOCITY_RATIO,
    MAP_PATH,
    PARAMS_PATH,
    STEPS_COUNT,
    SWARM_SIZE,
)
from swarm import Swarm
from w_solution import PathFinder


def main() -> None:
    finder = PathFinder.from_files(LANDSCAPES_PATH, MAP_PATH, PARAMS_PATH)

    swarm = Swarm(
        eval_fn=finder.get_W,
        swarm_size=SWARM_SIZE,
        neighbor_min=15,
        neighbor_max=25,
        max_iterations=STEPS_COUNT,
        inertia=CURRENT_VELOCITY_RATIO,
        cognitive=LOCAL_VELOCITY_RATIO,
        social=GLOBAL_VELOCITY_RATIO,
    )

    best_w = swarm.optimize()
    alpha, beta, _ = swarm.get_best_solution()

    print(f"Best W            = {best_w:.6f}")
    print(f"Std deviation     = {swarm.get_standard_deviation():.6f}")
    print(f"Best alpha        = {[round(a, 4) for a in alpha]}")
    print(f"Best beta         = {beta:.4f}")

    path = finder.get_path(alpha, beta)
    steps = " -> ".join(str(p) for p in path)
    print(f"Path ({len(path)} cells)  = {steps}")

    if path and path[-1] == finder.end:
        print("Goal reached.")
    else:
        print(f"Goal NOT reached (got stuck at {path[-1] if path else 'start'}).")


if __name__ == "__main__":
    main()
