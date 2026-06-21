import pytest
from particle import Particle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_particle(eval_fn=None, num_alpha=3):
    if eval_fn is None:
        eval_fn = lambda alpha, beta: sum(alpha) + beta
    return Particle(eval_fn, num_alpha=num_alpha,
                    alpha_bounds=(0.0, 1.0), beta_bounds=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParticleInit:
    def test_position_length(self):
        p = make_particle(num_alpha=5)
        assert len(p.position) == 6  # 5 alpha + 1 beta

    def test_velocity_length(self):
        p = make_particle(num_alpha=5)
        assert len(p.velocity) == 6

    def test_velocity_starts_at_zero(self):
        p = make_particle()
        assert all(v == 0.0 for v in p.velocity)

    def test_position_within_bounds(self):
        for _ in range(20):
            p = make_particle(num_alpha=10)
            for v in p.position:
                assert 0.0 <= v <= 1.0

    def test_best_w_set_after_init(self):
        p = make_particle()
        assert p.best_w != float("inf")

    def test_best_position_is_copy(self):
        p = make_particle()
        original_best = p.best_position.copy()
        p.position[0] = 999.0
        # best_position must not change with position
        assert p.best_position[0] == original_best[0]


class TestParticleEvaluate:
    def test_evaluate_updates_current_w(self):
        p = make_particle(eval_fn=lambda a, b: 0.0)
        p.current_w = float("inf")
        p.evaluate()
        assert p.current_w == 0.0

    def test_evaluate_updates_best_when_improved(self):
        call_count = [0]

        def descending(alpha, beta):
            call_count[0] += 1
            return 100.0 - call_count[0]

        p = make_particle(eval_fn=descending)
        for _ in range(5):
            p.evaluate()

        assert p.best_w < 100.0

    def test_best_position_not_updated_when_no_improvement(self):
        p = make_particle(eval_fn=lambda a, b: 1.0)
        # Force best already set to 1.0
        p.best_w = 0.0
        p.best_position = [99.0, 99.0, 99.0, 99.0]
        p.evaluate()
        # best_position must remain unchanged
        assert p.best_position == [99.0, 99.0, 99.0, 99.0]

    def test_best_position_is_copy_after_update(self):
        p = make_particle(eval_fn=lambda a, b: 0.0)
        before = p.best_position.copy()
        p.position[0] = 0.9999
        p.evaluate()
        new_best = p.best_position.copy()
        p.position[0] = -100.0
        assert p.best_position[0] == new_best[0]


class TestParticleUpdateVelocity:
    def test_position_changes_after_update(self):
        p = make_particle()
        pos_before = p.position.copy()
        global_best = [0.5] * len(p.position)
        p.update_velocity(global_best, inertia=0.5, cognitive=1.0, social=1.0)
        assert p.position != pos_before or all(v == 0.0 for v in p.velocity)

    def test_position_stays_within_bounds(self):
        for _ in range(30):
            p = make_particle()
            global_best = [1.0] * len(p.position)
            for _ in range(10):
                p.update_velocity(global_best, inertia=2.0, cognitive=2.0, social=2.0)
            for v in p.position:
                assert 0.0 <= v <= 1.0

    def test_velocity_all_dims_updated(self):
        p = make_particle(num_alpha=3)
        global_best = [1.0] * len(p.position)
        p.update_velocity(global_best, inertia=1.0, cognitive=1.0, social=1.0)
        # With a non-zero pull toward global_best and initial velocity=0,
        # at least some velocities should change
        assert any(v != 0.0 for v in p.velocity)

    def test_zero_inertia_zero_cognitive_only_social(self):
        p = make_particle(eval_fn=lambda a, b: 0.0)
        p.velocity = [0.0] * len(p.position)
        p.position = [0.0] * len(p.position)
        global_best = [1.0] * len(p.position)
        p.update_velocity(global_best, inertia=0.0, cognitive=0.0, social=1.0)
        # All positions should now equal r2 * (1.0 - 0.0) clamped to [0,1]
        for v in p.position:
            assert 0.0 <= v <= 1.0
