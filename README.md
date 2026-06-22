# PSO Agent Navigation

> 🇬🇧 English | [🇷🇺 Русский](README.ru.md)

A **Particle Swarm Optimisation (PSO)** system that finds the optimal navigation parameters for a greedy agent moving across a typed terrain grid.

## Features

- Greedy agent navigation across a square terrain grid using the heuristic `F(r, t, α, β) = α[t] + β / r`
- PSO-based optimisation of α (terrain weights) and β (proximity weight) parameters
- Configurable swarm size, iteration count, and velocity coefficients
- Random terrain map and parameter generation utilities
- Standalone W calculation without optimisation
- Full test suite with pytest

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Algorithm | Particle Swarm Optimisation (PSO) |
| Testing | pytest |
| Data format | JSON |

## Requirements

- Python 3.9+
- pytest (for running tests)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd swarm-pathfinder

# Install test dependencies
pip install pytest
```

The optimiser can then be launched with:

```bash
python main.py
```

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

## Running Tests

```bash
pytest tests/
```

The test suite covers utility functions, path-finder heuristics and navigation, individual particle behaviour, the full PSO loop, and data-generation functions.

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

## License

[MIT](LICENSE)
