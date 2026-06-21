from __future__ import annotations

import json
from typing import List, Tuple

from settings import LANDSCAPES_PATH, MAP_PATH, PARAMS_PATH, MAP_SIZE


class PathFinder:
    """Greedy agent that navigates a terrain grid using a heuristic function.

    The transition value F(r, t, alpha, beta) = alpha[t] + beta / r guides
    the agent at each step: it moves to the unvisited neighbour with the
    highest F.  W is the sum of F values accumulated along the route and is
    the objective that PSO minimises.
    """

    def __init__(
        self,
        landscapes: dict,
        terrain_map: List[List[int]],
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> None:
        self.landscapes = landscapes
        self.terrain_map = terrain_map
        self.start = tuple(start)
        self.end = tuple(end)
        self.map_rows = len(terrain_map)
        self.map_cols = len(terrain_map[0]) if terrain_map else 0

    @classmethod
    def from_files(
        cls,
        landscapes_path: str = LANDSCAPES_PATH,
        map_path: str = MAP_PATH,
        params_path: str = PARAMS_PATH,
    ) -> "PathFinder":
        with open(landscapes_path) as f:
            landscapes_raw = json.loads(f.read())
        # JSON keys are always strings; convert to int for indexed access
        landscapes = {int(k): v for k, v in landscapes_raw.items()}

        with open(map_path) as f:
            terrain_map = json.loads(f.read())

        with open(params_path) as f:
            params = json.loads(f.read())

        return cls(landscapes, terrain_map, params["start"], params["end"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _manhattan_distance(
        self, point1: Tuple[int, int], point2: Tuple[int, int]
    ) -> int:
        return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])

    def _transition_value(
        self, r: int, terrain_type: int, alpha: List[float], beta: float
    ) -> float:
        """F = alpha[terrain_type] + beta / r."""
        return alpha[terrain_type] + beta / r

    def _get_neighbors(self, point: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = point
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if x < self.map_rows - 1:
            neighbors.append((x + 1, y))
        if y < self.map_cols - 1:
            neighbors.append((x, y + 1))
        return neighbors

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_W(self, alpha: List[float], beta: float) -> float:
        """Navigate from start to end and return the accumulated W value."""
        visited = []
        current = self.start
        visited.append(current)
        w = 0.0

        while current != self.end:
            x, y = current
            terrain_type = self.terrain_map[x][y]
            neighbors = self._get_neighbors(current)

            best_f = float("-inf")
            next_point = None

            for point in neighbors:
                if point in visited:
                    continue
                r = self._manhattan_distance(point, self.end)
                f = float("inf") if r == 0 else self._transition_value(r, terrain_type, alpha, beta)
                if f > best_f:
                    best_f = f
                    next_point = point

            if next_point is None:
                # All neighbours exhausted without reaching the goal
                break

            visited.append(next_point)
            current = next_point

            if best_f not in (float("inf"), float("-inf")):
                w += best_f

        return w

    def get_path(self, alpha: List[float], beta: float) -> List[Tuple[int, int]]:
        """Return the full sequence of grid cells visited from start to end."""
        visited = []
        current = self.start
        visited.append(current)

        while current != self.end:
            x, y = current
            terrain_type = self.terrain_map[x][y]
            neighbors = self._get_neighbors(current)

            best_f = float("-inf")
            next_point = None

            for point in neighbors:
                if point in visited:
                    continue
                r = self._manhattan_distance(point, self.end)
                f = float("inf") if r == 0 else self._transition_value(r, terrain_type, alpha, beta)
                if f > best_f:
                    best_f = f
                    next_point = point

            if next_point is None:
                break

            visited.append(next_point)
            current = next_point

        return visited


# ---------------------------------------------------------------------------
# Module-level backward-compatible API
# ---------------------------------------------------------------------------

_default_finder: "PathFinder | None" = None


def _get_default_finder() -> PathFinder:
    global _default_finder
    if _default_finder is None:
        _default_finder = PathFinder.from_files()
    return _default_finder


def get_W(alpha: List[float], beta: float) -> float:
    """Module-level wrapper kept for backward compatibility."""
    return _get_default_finder().get_W(alpha, beta)


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json
    from settings import PARAMS_PATH as _PARAMS_PATH

    with open(_PARAMS_PATH) as _f:
        _params = _json.loads(_f.read())

    _finder = PathFinder.from_files()
    _W = _finder.get_W(_params["a"], _params["b"])
    print(f"W for initial parameters = {_W}")
