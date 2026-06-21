"""Generate random input data: terrain types, map, and algorithm parameters."""
from __future__ import annotations

import json
from os.path import exists
from os import remove
from random import randint, choice
from shutil import copy
from typing import Dict, List, Optional, Tuple

from settings import (
    ALPHA_MAX,
    ALPHA_MIN,
    BETA_MAX,
    BETA_MIN,
    LANDSCAPE_COST_MAX,
    LANDSCAPE_COST_MIN,
    LANDSCAPES_COUNT,
    LANDSCAPES_PATH,
    MAP_PATH,
    MAP_SIZE,
    PARAMS_PATH,
    WILL_BACKUP,
)
from utils import get_random


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_to_json(data: object, file_path: str) -> bool:
    try:
        if exists(file_path):
            if WILL_BACKUP:
                copy(file_path, file_path + ".backup")
            remove(file_path)

        with open(file_path, mode="x") as f:
            json.dump(data, f, indent=4)

        return True
    except Exception as exc:
        print(f"Could not save {file_path}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def generate_point(map_size: int) -> Tuple[int, int]:
    """Return a random (x, y) point with indices in [0, map_size - 1]."""
    return (randint(0, map_size - 1), randint(0, map_size - 1))


def generate_landscapes(
    count: int = LANDSCAPES_COUNT,
    cost_min: int = LANDSCAPE_COST_MIN,
    cost_max: int = LANDSCAPE_COST_MAX,
) -> Dict[int, int]:
    """Return a dict mapping terrain-type index → movement cost."""
    landscapes = {i: randint(cost_min, cost_max) for i in range(count)}

    if _save_to_json({str(k): v for k, v in landscapes.items()}, LANDSCAPES_PATH):
        print(f"{count} terrain types saved to {LANDSCAPES_PATH}")
    else:
        print("Failed to save terrain types.")

    return landscapes


def generate_map(
    map_size: int = MAP_SIZE,
    landscapes: Optional[Dict[int, int]] = None,
    num_terrain_types: int = LANDSCAPES_COUNT,
) -> List[List[int]]:
    """Return a map_size × map_size grid of terrain-type indices."""
    if landscapes is not None:
        terrain_types = list(landscapes.keys())
    else:
        terrain_types = list(range(num_terrain_types))

    terrain_map = [
        [choice(terrain_types) for _ in range(map_size)]
        for _ in range(map_size)
    ]

    if _save_to_json(terrain_map, MAP_PATH):
        print(f"{map_size}x{map_size} map saved to {MAP_PATH}")
    else:
        print("Failed to save map.")

    return terrain_map


def generate_params(
    landscapes_count: int = LANDSCAPES_COUNT,
    map_size: int = MAP_SIZE,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
    beta_min: float = BETA_MIN,
    beta_max: float = BETA_MAX,
) -> dict:
    """Return a dict with random alpha, beta, start, and end parameters."""
    params = {
        "a": [get_random(alpha_min, alpha_max) for _ in range(landscapes_count)],
        "b": get_random(beta_min, beta_max),
        "start": list(generate_point(map_size)),
        "end": list(generate_point(map_size)),
    }

    if _save_to_json(params, PARAMS_PATH):
        print(f"Parameters saved to {PARAMS_PATH}")
    else:
        print("Failed to save parameters.")

    return params


def generate() -> dict:
    """Generate and persist all input data; return generated values."""
    print()
    landscapes = generate_landscapes()
    terrain_map = generate_map(landscapes=landscapes)
    params = generate_params()
    return {"landscapes": landscapes, "map": terrain_map, "params": params}


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate()
