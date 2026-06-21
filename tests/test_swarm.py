import math
import pytest
from swarm import Swarm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_swarm(eval_fn, swarm_size=10, max_iterations=20, num_alpha=2):
    return Swarm(
        eval_fn=eval_fn,
        swarm_size=swarm_size,
        neighbor_min=3,
        neighbor_max=5,
        max_iterations=max_iterations,
        inertia=0.5,
        cognitive=1.0,
        social=1.0,
        num_alpha=num_alpha,
        alpha_bounds=(0.0, 1.0),
        beta_bounds=(0.0, 1.0),
    )


def sphere(alpha, beta):
    """Simple convex function; minimum = 0 at alpha=[0,...,0], beta=0."""
    return sum(a ** 2 for a in alpha) + beta ** 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSwarmInit:
    def test_correct_number_of_particles(self):
        swarm = make_swarm(sphere, swarm_size=15)
        assert len(swarm.particles) == 15

    def test_best_w_initialised(self):
        swarm = make_swarm(sphere)
        assert swarm.best_w < float("inf")

    def test_best_position_initialised(self):
        swarm = make_swarm(sphere)
        assert swarm.best_position is not None


class TestSwarmOptimize:
    def test_returns_float(self):
        swarm = make_swarm(sphere)
        result = swarm.optimize()
        assert isinstance(result, float)

    def test_result_non_negative(self):
        swarm = make_swarm(sphere)
        result = swarm.optimize()
        assert result >= 0.0

    def test_optimises_toward_minimum(self):
        swarm = make_swarm(sphere, swarm_size=30, max_iterations=100)
        best_w = swarm.optimize()
        # sphere minimum is 0; expect to get close within bounded [0,1]^3
        assert best_w < 0.5

    def test_convergence_history_length(self):
        iterations = 15
        swarm = make_swarm(sphere, max_iterations=iterations)
        swarm.optimize()
        assert len(swarm.get_convergence_history()) == iterations

    def test_convergence_history_non_increasing(self):
        swarm = make_swarm(sphere, swarm_size=20, max_iterations=50)
        swarm.optimize()
        history = swarm.get_convergence_history()
        for prev, curr in zip(history, history[1:]):
            assert curr <= prev + 1e-12  # allow tiny float noise


class TestGetBestSolution:
    def test_raises_before_optimize(self):
        swarm = make_swarm(sphere)
        swarm.best_position = None
        with pytest.raises(RuntimeError):
            swarm.get_best_solution()

    def test_returns_three_tuple(self):
        swarm = make_swarm(sphere, num_alpha=3)
        swarm.optimize()
        alpha, beta, best_w = swarm.get_best_solution()
        assert len(alpha) == 3
        assert isinstance(beta, float)
        assert isinstance(best_w, float)

    def test_alpha_within_bounds(self):
        swarm = make_swarm(sphere)
        swarm.optimize()
        alpha, beta, _ = swarm.get_best_solution()
        for a in alpha:
            assert 0.0 <= a <= 1.0
        assert 0.0 <= beta <= 1.0


class TestGetStandardDeviation:
    def test_returns_non_negative(self):
        swarm = make_swarm(sphere)
        swarm.optimize()
        assert swarm.get_standard_deviation() >= 0.0

    def test_constant_fn_gives_zero_deviation(self):
        swarm = make_swarm(lambda a, b: 1.0)
        swarm.optimize()
        assert swarm.get_standard_deviation() == pytest.approx(0.0, abs=1e-9)


class TestGetConvergenceHistory:
    def test_returns_copy(self):
        swarm = make_swarm(sphere, max_iterations=5)
        swarm.optimize()
        hist = swarm.get_convergence_history()
        hist.append(999.0)
        assert len(swarm.get_convergence_history()) == 5

    def test_empty_before_optimize(self):
        swarm = make_swarm(sphere)
        assert swarm.get_convergence_history() == []
