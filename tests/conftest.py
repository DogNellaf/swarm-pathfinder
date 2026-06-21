"""Shared pytest fixtures."""
import sys
import os

# Ensure the project root is on the path so imports work from the tests/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from w_solution import PathFinder


@pytest.fixture()
def simple_landscapes() -> dict:
    """3 terrain types with integer costs."""
    return {0: 1, 1: 5, 2: 10}


@pytest.fixture()
def simple_map() -> list:
    """3x3 terrain map with type indices 0-2."""
    return [
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0],
    ]


@pytest.fixture()
def finder(simple_landscapes, simple_map) -> PathFinder:
    """PathFinder on a 3x3 grid, start=(0,0), end=(2,2)."""
    return PathFinder(
        landscapes=simple_landscapes,
        terrain_map=simple_map,
        start=(0, 0),
        end=(2, 2),
    )


@pytest.fixture()
def finder_same_point(simple_landscapes, simple_map) -> PathFinder:
    """PathFinder where start == end."""
    return PathFinder(
        landscapes=simple_landscapes,
        terrain_map=simple_map,
        start=(1, 1),
        end=(1, 1),
    )


@pytest.fixture()
def uniform_alpha() -> list:
    """Alpha vector with equal weight for all terrain types."""
    return [0.5, 0.5, 0.5]


@pytest.fixture()
def uniform_beta() -> float:
    return 1.0


@pytest.fixture()
def trivial_eval_fn():
    """Evaluation function that always returns a constant."""
    return lambda alpha, beta: 42.0
