import pytest
from utils import clamp, get_random


class TestGetRandom:
    def test_within_default_bounds(self):
        for _ in range(100):
            v = get_random()
            assert 0.0 <= v <= 1.0

    def test_within_custom_bounds(self):
        for _ in range(100):
            v = get_random(2.0, 5.0)
            assert 2.0 <= v <= 5.0

    def test_equal_bounds_returns_that_value(self):
        assert get_random(3.0, 3.0) == 3.0

    def test_negative_bounds(self):
        for _ in range(50):
            v = get_random(-10.0, -1.0)
            assert -10.0 <= v <= -1.0

    def test_returns_float(self):
        assert isinstance(get_random(), float)


class TestClamp:
    def test_value_below_min(self):
        assert clamp(-5.0, 0.0, 1.0) == 0.0

    def test_value_above_max(self):
        assert clamp(10.0, 0.0, 1.0) == 1.0

    def test_value_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_value_at_min(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0

    def test_value_at_max(self):
        assert clamp(1.0, 0.0, 1.0) == 1.0

    def test_integer_values(self):
        assert clamp(15, 0, 10) == 10
        assert clamp(-3, 0, 10) == 0
        assert clamp(5, 0, 10) == 5
