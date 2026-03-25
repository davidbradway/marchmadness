# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

March Madness bracket scoring system: reads 40 user bracket CSV files, compares predictions to actual tournament results, outputs ranked standings, and simulates all possible remaining outcomes to project final placements.

## Scripts

### `score_brackets.py` — Current Standings

```bash
python score_brackets.py
```

Uses Conda environment (configured in `.vscode/settings.json`). If using Conda explicitly:

```bash
conda run python score_brackets.py
```

No external dependencies — standard library only (`csv`, `os`, `glob`).

### `simulate_outcomes.py` — Outcome Simulation

```bash
python simulate_outcomes.py
```

Requires `numpy`. Uses Conda environment.

## Architecture

### `score_brackets.py`

Single-script ETL pipeline:

1. **Load** `group_scoring.csv` → round name → point value mapping
2. **Load** `winners.csv` → `(Round, Winner)` lookup dict for actual results
3. **Score** each `brackets/*.csv` via glob — accumulate points for matching predictions
4. **Output** `standings.csv` (ranked results) and print leaderboard to console

### `simulate_outcomes.py`

Exhaustive scenario simulation using NumPy vectorization:

1. **Load** current scores from all brackets (same logic as `score_brackets.py`)
2. **Define** the 15 future games: Sweet 16 (8), Elite 8 (4), Final Four (2), Championship (1)
3. **Enumerate** all 2^15 = 32,768 possible outcomes as a bit matrix
4. **Parse** each bracket's predictions for all 15 future games
5. **Score** all scenarios × brackets simultaneously via matrix operations
6. **Rank** brackets within each scenario (ties share the lowest rank)
7. **Output** placement counts, expected rank, and best-case finish per bracket to console and `simulation_results.csv`

## Data Files

| File | Purpose |
|------|---------|
| `group_scoring.csv` | Points per round: 1, 2, 4, 8, 16, 32 (Round of 64 → National Final) |
| `winners.csv` | Actual tournament results (63 games) |
| `brackets/*.csv` | User prediction files (40 total) — same schema as `winners.csv` with a "Predicted Winner" column |
| `standings.csv` | Generated output — rank, filename, score |
| `simulation_results.csv` | Generated output — expected rank and placement counts across all 32,768 scenarios |

Adding a new participant: drop a new CSV file in the `brackets/` directory and re-run the scripts.
