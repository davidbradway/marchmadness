# March Madness Bracket Scoring

Reads 40 user bracket CSV files, compares predictions to actual tournament results, outputs ranked standings, and simulates all possible remaining outcomes to project final placements.

## Scripts

### `score_brackets.py` — Current Standings

Scores all brackets against confirmed results and writes a leaderboard.

```bash
python score_brackets.py
```

Output: `standings.csv` + console leaderboard.

No external dependencies — standard library only (`csv`, `os`, `glob`).

### `simulate_outcomes.py` — Outcome Simulation

Enumerates all possible outcomes for the remaining games (Sweet 16 through Championship), scores every bracket in every scenario, and reports each participant's probability of finishing 1st or 2nd, best-case finish, and expected rank. Scenarios are weighted by likelihood using a seed-based win-probability model: P(A beats B) = seed_B^0.8 / (seed_A^0.8 + seed_B^0.8).

```bash
python simulate_outcomes.py
```

Output: `simulation_results.csv` + console summary table.

Requires `numpy`.

## Data Files

| File | Purpose |
|------|---------|
| `group_scoring.csv` | Points per round: 1, 2, 4, 8, 16, 32 (Round of 64 → National Final) |
| `winners.csv` | Actual tournament results (63 games) |
| `brackets/*.csv` | User prediction files (40 total) |
| `standings.csv` | Generated — rank, bracket name, score |
| `simulation_results.csv` | Generated — expected rank and probability-weighted placement probabilities across all scenarios |

## Adding a Participant

Drop a new CSV file in the `brackets/` directory (same schema as existing bracket files) and re-run the scripts.
