# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

March Madness bracket scoring system: reads 40 user bracket CSV files, compares predictions to actual tournament results, and outputs ranked standings.

## Running the Script

```bash
python score_brackets.py
```

Uses Conda environment (configured in `.vscode/settings.json`). If using Conda explicitly:

```bash
conda run python score_brackets.py
```

No external dependencies — standard library only (`csv`, `os`, `glob`).

## Architecture

Single-script ETL pipeline (`score_brackets.py`):

1. **Load** `group_scoring.csv` → round name → point value mapping
2. **Load** `winners.csv` → `(Round, Winner)` lookup dict for actual results
3. **Score** each `bracket_*.csv` via glob — accumulate points for matching predictions
4. **Output** `standings.csv` (ranked results) and print leaderboard to console

## Data Files

| File | Purpose |
|------|---------|
| `group_scoring.csv` | Points per round: 1, 2, 4, 8, 16, 32 (Round of 64 → National Final) |
| `winners.csv` | Actual tournament results (63 games) |
| `bracket_*.csv` | User prediction files (40 total) — same schema as `winners.csv` with a "Predicted Winner" column |
| `standings.csv` | Generated output — rank, filename, score |

Adding a new participant: drop a new `bracket_<name>.csv` file and re-run the script.
