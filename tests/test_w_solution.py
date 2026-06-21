import pytest
from w_solution import PathFinder


class TestManhattanDistance:
    def test_same_point(self, finder):
        assert finder._manhattan_distance((0, 0), (0, 0)) == 0

    def test_horizontal(self, finder):
        assert finder._manhattan_distance((0, 0), (0, 3)) == 3

    def test_vertical(self, finder):
        assert finder._manhattan_distance((0, 0), (4, 0)) == 4

    def test_diagonal(self, finder):
        assert finder._manhattan_distance((0, 0), (2, 2)) == 4

    def test_negative_direction(self, finder):
        assert finder._manhattan_distance((3, 3), (0, 0)) == 6


class TestTransitionValue:
    def test_basic_formula(self, finder):
        # F = alpha[t] + beta/r
        result = finder._transition_value(r=2, terrain_type=1, alpha=[0.1, 0.5, 0.9], beta=1.0)
        assert result == pytest.approx(0.5 + 1.0 / 2)

    def test_beta_contribution(self, finder):
        # larger beta → larger F
        f1 = finder._transition_value(2, 0, [1.0, 0.0, 0.0], 0.0)
        f2 = finder._transition_value(2, 0, [1.0, 0.0, 0.0], 2.0)
        assert f2 > f1

    def test_closer_distance_increases_f(self, finder):
        alpha = [0.5, 0.5, 0.5]
        f_close = finder._transition_value(1, 0, alpha, 1.0)
        f_far = finder._transition_value(5, 0, alpha, 1.0)
        assert f_close > f_far


class TestGetNeighbors:
    def test_corner_top_left(self, finder):
        # 3x3 grid, (0,0) has 2 neighbours
        nbrs = finder._get_neighbors((0, 0))
        assert (1, 0) in nbrs
        assert (0, 1) in nbrs
        assert len(nbrs) == 2

    def test_corner_bottom_right(self, finder):
        nbrs = finder._get_neighbors((2, 2))
        assert (1, 2) in nbrs
        assert (2, 1) in nbrs
        assert len(nbrs) == 2

    def test_center(self, finder):
        nbrs = finder._get_neighbors((1, 1))
        assert len(nbrs) == 4
        assert (0, 1) in nbrs
        assert (2, 1) in nbrs
        assert (1, 0) in nbrs
        assert (1, 2) in nbrs

    def test_edge(self, finder):
        nbrs = finder._get_neighbors((0, 1))
        assert len(nbrs) == 3


class TestGetW:
    def test_start_equals_end_returns_zero(self, finder_same_point, uniform_alpha, uniform_beta):
        w = finder_same_point.get_W(uniform_alpha, uniform_beta)
        assert w == 0.0

    def test_returns_finite_positive_value(self, finder, uniform_alpha, uniform_beta):
        w = finder.get_W(uniform_alpha, uniform_beta)
        assert w > 0.0
        assert w != float("inf")

    def test_higher_beta_changes_w(self, finder, uniform_alpha):
        w1 = finder.get_W(uniform_alpha, 0.1)
        w2 = finder.get_W(uniform_alpha, 5.0)
        assert w1 != w2

    def test_all_zero_alpha_still_navigates(self, finder, uniform_beta):
        zero_alpha = [0.0, 0.0, 0.0]
        w = finder.get_W(zero_alpha, uniform_beta)
        # With only beta/r as heuristic the agent still moves; result must be finite
        assert w != float("inf")


class TestGetPath:
    def test_starts_at_start(self, finder, uniform_alpha, uniform_beta):
        path = finder.get_path(uniform_alpha, uniform_beta)
        assert path[0] == finder.start

    def test_ends_at_end_when_reachable(self, finder, uniform_alpha, uniform_beta):
        path = finder.get_path(uniform_alpha, uniform_beta)
        assert path[-1] == finder.end

    def test_no_repeated_cells(self, finder, uniform_alpha, uniform_beta):
        path = finder.get_path(uniform_alpha, uniform_beta)
        assert len(path) == len(set(path))

    def test_start_equals_end(self, finder_same_point, uniform_alpha, uniform_beta):
        path = finder_same_point.get_path(uniform_alpha, uniform_beta)
        assert path == [finder_same_point.start]

    def test_consecutive_cells_are_neighbors(self, finder, uniform_alpha, uniform_beta):
        path = finder.get_path(uniform_alpha, uniform_beta)
        for a, b in zip(path, path[1:]):
            dist = finder._manhattan_distance(a, b)
            assert dist == 1, f"Non-adjacent step {a} -> {b}"
