# PSO Agent Navigation

> [English](README.md) | [Русский](README.ru.md)

A **Particle Swarm Optimisation (PSO)** system that finds the optimal navigation parameters for a greedy agent moving across a typed terrain grid.

---

## Overview

The agent navigates from a start cell to a goal cell on a square grid.  Each cell belongs to one of *N* terrain types, and the agent chooses its next step greedily using the transition heuristic

```
F(r, t, α, β) = α[t] + β / r
```

where

| Symbol | Meaning |
|--------|---------|
| `r` | Manhattan distance from the candidate cell to the goal |
| `t` | Terrain type of the **current** cell |
| `α[t]` | Weight assigned to terrain type `t` |
| `β` | Weight assigned to proximity to goal |

At every cell the agent moves to the unvisited neighbour with the highest `F`.  The objective `W` is the sum of `F` values accumulated along the route.  PSO minimises `W` by searching for the best `α` vector and `β` scalar.

---

## Project Structure

```
.
├── settings.py          # Global constants and file paths
├── utils.py             # get_random(), clamp()
├── w_solution.py        # PathFinder class — agent simulation and W calculation
├── particle.py          # Particle class — single PSO agent
├── swarm.py             # Swarm class — PSO loop
├── swarp.py             # Backward-compatible alias for Swarm (legacy notebooks)
├── input_generator.py   # Random data generation utilities
├── main.py              # Entry point
├── data/
│   ├── landscapes.json  # Terrain-type → movement cost mapping
│   ├── map.json         # Grid of terrain-type indices
│   └── params.json      # Initial α, β, start, end
└── tests/
    ├── conftest.py
    ├── test_utils.py
    ├── test_w_solution.py
    ├── test_particle.py
    ├── test_swarm.py
    └── test_input_generator.py
```

---

## Installation

Python 3.9+ is required.  No third-party dependencies are needed for the core algorithm.  `pytest` is required to run the tests.

```bash
pip install pytest
```

---

## Usage

### Run the optimiser

```bash
python main.py
```

The script loads the data files from `data/`, runs PSO, and prints the best `W`, the standard deviation of the swarm, the optimised `α` and `β`, and the resulting path.

### Generate new random data

```bash
python input_generator.py
```

### Calculate W for the stored parameters without optimisation

```bash
python w_solution.py
```

### Run PSO directly

```bash
python swarm.py
```

---

## Configuration

All tuneable constants live in [settings.py](settings.py):

| Constant | Default | Description |
|----------|---------|-------------|
| `MAP_SIZE` | `10` | Side length of the square grid |
| `LANDSCAPES_COUNT` | `10` | Number of terrain types |
| `SWARM_SIZE` | `200` | Number of PSO particles |
| `STEPS_COUNT` | `1000` | PSO iterations |
| `CURRENT_VELOCITY_RATIO` | `0.1` | Inertia weight `w` |
| `LOCAL_VELOCITY_RATIO` | `1.0` | Cognitive coefficient `c1` |
| `GLOBAL_VELOCITY_RATIO` | `5.0` | Social coefficient `c2` |
| `ALPHA_MIN / ALPHA_MAX` | `0 / 1` | Search bounds for each `α` |
| `BETA_MIN / BETA_MAX` | `0 / 1` | Search bounds for `β` |
| `WILL_BACKUP` | `False` | Keep `.backup` copy when regenerating data |

---

## Running Tests

```bash
pytest tests/
```

The test suite covers utility functions, the path-finder heuristics and navigation, individual particle behaviour, the full PSO loop, and data-generation functions.

---

## Algorithm Details

### PSO velocity update (standard form)

```
v(t+1) = w · v(t)
        + c1 · r1 · (pbest − x(t))
        + c2 · r2 · (gbest_informant − x(t))
```

Each particle selects a random *neighbourhood* of `[neighbor_min, neighbor_max]` particles, identifies the informant with the best personal-best `W`, and updates its velocity accordingly.  After the velocity update the particle evaluates its new position so that learning actually occurs each iteration.

The search space is `LANDSCAPES_COUNT + 1` dimensional: one `α` value per terrain type and one `β` value.  All coordinates are clamped to `[0, 1]` after each velocity update.

---

## License

MIT License