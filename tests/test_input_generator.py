import json
import os
import tempfile
import pytest

from input_generator import (
    generate_landscapes,
    generate_map,
    generate_params,
    generate_point,
)
from settings import (
    ALPHA_MAX,
    ALPHA_MIN,
    BETA_MAX,
    BETA_MIN,
    LANDSCAPE_COST_MAX,
    LANDSCAPE_COST_MIN,
    LANDSCAPES_COUNT,
    MAP_SIZE,
)


class TestGeneratePoint:
    def test_within_bounds(self):
        for _ in range(200):
            x, y = generate_point(MAP_SIZE)
            assert 0 <= x <= MAP_SIZE - 1
            assert 0 <= y <= MAP_SIZE - 1

    def test_returns_tuple_of_two(self):
        pt = generate_point(10)
        assert len(pt) == 2

    def test_small_map_size(self):
        for _ in range(50):
            x, y = generate_point(1)
            assert x == 0 and y == 0


class TestGenerateLandscapes:
    def test_correct_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "input_generator.LANDSCAPES_PATH", str(tmp_path / "ls.json")
        )
        result = generate_landscapes(count=5)
        assert len(result) == 5

    def test_keys_are_sequential(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "input_generator.LANDSCAPES_PATH", str(tmp_path / "ls.json")
        )
        result = generate_landscapes(count=4)
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_costs_within_bounds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "input_generator.LANDSCAPES_PATH", str(tmp_path / "ls.json")
        )
        result = generate_landscapes()
        for cost in result.values():
            assert LANDSCAPE_COST_MIN <= cost <= LANDSCAPE_COST_MAX

    def test_file_is_valid_json(self, tmp_path, monkeypatch):
        dest = str(tmp_path / "ls.json")
        monkeypatch.setattr("input_generator.LANDSCAPES_PATH", dest)
        generate_landscapes()
        with open(dest) as f:
            data = json.loads(f.read())
        assert isinstance(data, dict)


class TestGenerateMap:
    def test_dimensions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("input_generator.MAP_PATH", str(tmp_path / "map.json"))
        result = generate_map(map_size=5, num_terrain_types=3)
        assert len(result) == 5
        assert all(len(row) == 5 for row in result)

    def test_values_are_valid_terrain_types(self, tmp_path, monkeypatch):
        monkeypatch.setattr("input_generator.MAP_PATH", str(tmp_path / "map.json"))
        landscapes = {0: 1, 1: 2, 2: 3}
        result = generate_map(map_size=4, landscapes=landscapes)
        for row in result:
            for cell in row:
                assert cell in landscapes

    def test_file_is_valid_json(self, tmp_path, monkeypatch):
        dest = str(tmp_path / "map.json")
        monkeypatch.setattr("input_generator.MAP_PATH", dest)
        generate_map()
        with open(dest) as f:
            data = json.loads(f.read())
        assert isinstance(data, list)


class TestGenerateParams:
    def test_alpha_length(self, tmp_path, monkeypatch):
        monkeypatch.setattr("input_generator.PARAMS_PATH", str(tmp_path / "p.json"))
        result = generate_params(landscapes_count=7)
        assert len(result["a"]) == 7

    def test_alpha_within_bounds(self, tmp_path, monkeypatch):
        monkeypatch.setattr("input_generator.PARAMS_PATH", str(tmp_path / "p.json"))
        result = generate_params()
        for a in result["a"]:
            assert ALPHA_MIN <= a <= ALPHA_MAX

    def test_beta_within_bounds(self, tmp_path, monkeypatch):
        monkeypatch.setattr("input_generator.PARAMS_PATH", str(tmp_path / "p.json"))
        result = generate_params()
        assert BETA_MIN <= result["b"] <= BETA_MAX

    def test_start_end_within_map(self, tmp_path, monkeypatch):
        monkeypatch.setattr("input_generator.PARAMS_PATH", str(tmp_path / "p.json"))
        size = 8
        for _ in range(30):
            result = generate_params(map_size=size)
            for coord in result["start"] + result["end"]:
                assert 0 <= coord <= size - 1

    def test_file_is_valid_json(self, tmp_path, monkeypatch):
        dest = str(tmp_path / "p.json")
        monkeypatch.setattr("input_generator.PARAMS_PATH", dest)
        generate_params()
        with open(dest) as f:
            data = json.loads(f.read())
        assert "a" in data and "b" in data and "start" in data and "end" in data
