import csv
import glob
import os
import numpy as np

from tournament import load_point_values, load_actual_winners, load_seed_map, load_all_scores

# ---------------------------------------------------------------------------
# 1. Load current (already-scored) state
# ---------------------------------------------------------------------------

point_values   = load_point_values()
actual_winners = load_actual_winners()

bracket_files = sorted(glob.glob('brackets/*.csv'))
bracket_names = [os.path.basename(p) for p in bracket_files]
N = len(bracket_files)

current_scores = np.array(
    load_all_scores(bracket_files, actual_winners, point_values),
    dtype=np.int32,
)  # (N,)

# ---------------------------------------------------------------------------
# 2. Define the 3 remaining future games
#
# Final Four is fully set after all Elite Eight results:
#   East/South FF:   Connecticut (2) vs Illinois (3)
#   West/Midwest FF: Arizona (1)     vs Michigan (1)
#
# Games still to play:
#   Game 0:  Final Four (East/South)   — Connecticut vs Illinois
#   Game 1:  Final Four (West/Midwest) — Arizona vs Michigan
#   Game 2:  Championship              — winner of game 0 vs winner of game 1
#
# Total: 3 games → 2^3 = 8 scenarios
# ---------------------------------------------------------------------------

# Team IDs: 0=Connecticut, 1=Illinois, 2=Arizona, 3=Michigan
FINAL_FOUR = [
    (0, "East/South",   "Round of 4", "Connecticut", "Illinois"),
    (1, "West/Midwest", "Round of 4", "Arizona",     "Michigan"),
]

all_teams = ["Connecticut", "Illinois", "Arizona", "Michigan"]
team_to_id = {t: i for i, t in enumerate(all_teams)}

game_points = np.array(
    [point_values["Round of 4"]]     * 2 +   # games 0–1: Final Four
    [point_values["National Final"]] * 1,    # game 2:    Championship
    dtype=np.int32,
)  # (3,)

# ---------------------------------------------------------------------------
# 2.5. Win-probability matrix (seed-based)
#
#   Seeds read from winners.csv (Round of 64 rows).
#   P(A beats B) = seed_B^k / (seed_A^k + seed_B^k),  k = 0.80
# ---------------------------------------------------------------------------

SEED_K = 0.80

seed_map = load_seed_map()

seeds = np.array(
    [seed_map.get(t, 8) for t in all_teams], dtype=np.float64
)
s_pow = seeds ** SEED_K
prob_matrix = s_pow[np.newaxis, :] / (s_pow[:, np.newaxis] + s_pow[np.newaxis, :])
np.fill_diagonal(prob_matrix, 0.5)

print(f"Win probabilities  [seed-based model, k={SEED_K}]:")
print(f"  {'Matchup':<8} {'Team 1':<16} vs {'Team 2':<16}  P(T1 wins)")
print(f"  {'-'*57}")
for _, label, _, t1, t2 in FINAL_FOUR:
    p = prob_matrix[team_to_id[t1], team_to_id[t2]]
    print(f"  FF       {t1:<16}    {t2:<16}  {p:.1%}")

# ---------------------------------------------------------------------------
# 3. Enumerate all 2^3 = 8 scenarios
# ---------------------------------------------------------------------------

bits = ((np.arange(8, dtype=np.int32)[:, np.newaxis]) >> np.arange(3)) & 1
# shape: (8, 3)

# Final Four winners (games 0–1)
ff_t1 = np.array([0, 2], dtype=np.int32)   # Connecticut, Arizona
ff_t2 = np.array([1, 3], dtype=np.int32)   # Illinois, Michigan
ff_winners = np.where(bits[:, :2] == 0, ff_t1, ff_t2)  # (8, 2)
# column 0 = East/South winner, column 1 = West/Midwest winner

# Championship (game 2)
champ_winner = np.where(bits[:, 2:3] == 0, ff_winners[:, 0:1], ff_winners[:, 1:2])  # (8, 1)

all_winners = np.concatenate([ff_winners, champ_winner], axis=1)  # (8, 3)

# ---------------------------------------------------------------------------
# 3.5. Scenario weights
# ---------------------------------------------------------------------------

ff_p1  = prob_matrix[ff_t1, ff_t2]                                            # (2,)
ff_out = np.where(bits[:, :2] == 0, ff_p1[np.newaxis, :], 1.0 - ff_p1)       # (8, 2)

champ_p1  = prob_matrix[ff_winners[:, 0:1], ff_winners[:, 1:2]]               # (8, 1)
champ_out = np.where(bits[:, 2:3] == 0, champ_p1, 1.0 - champ_p1)            # (8, 1)

all_game_probs = np.concatenate([ff_out, champ_out], axis=1)  # (8, 3)

scenario_weights = all_game_probs.prod(axis=1)
scenario_weights /= scenario_weights.sum()

# ---------------------------------------------------------------------------
# 4. Parse each bracket's predictions for the 3 future games
# ---------------------------------------------------------------------------

def get_future_predictions(path):
    """Return a length-3 list of team IDs (or -1 if team already eliminated)."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    lookup = {}
    ff_preds = []
    for row in rows:
        region = row['Region'].strip()
        rnd    = row['Round'].strip()
        winner = row['Predicted Winner'].strip()
        if region == 'Final Four' and rnd == 'Round of 4':
            ff_preds.append(winner)  # order: East/South first, West/Midwest second
        else:
            lookup[(region, rnd)] = winner

    preds = []

    # Games 0–1: Final Four (East/South first, West/Midwest second)
    for team in (ff_preds + ['', ''])[:2]:
        preds.append(team_to_id.get(team, -1))

    # Game 2: Championship
    team = lookup.get(('Championship', 'National Final'), '')
    preds.append(team_to_id.get(team, -1))

    return preds

bracket_preds = np.array(
    [get_future_predictions(p) for p in bracket_files],
    dtype=np.int32,
)  # (N, 3)

# ---------------------------------------------------------------------------
# 5. Score all scenarios × brackets simultaneously
# ---------------------------------------------------------------------------

matches = (
    all_winners[:, np.newaxis, :] == bracket_preds[np.newaxis, :, :]
)  # (8, N, 3)

valid = (bracket_preds != -1)[np.newaxis, :, :]
matches &= valid

future_scores = (matches * game_points).sum(axis=2).astype(np.int32)   # (8, N)
total_scores  = current_scores[np.newaxis, :] + future_scores           # (8, N)

# ---------------------------------------------------------------------------
# 6. Rank brackets in each scenario
# ---------------------------------------------------------------------------

rank_matrix = (
    1 + (total_scores[:, :, np.newaxis] > total_scores[:, np.newaxis, :]).sum(axis=1)
)  # (8, N)

# ---------------------------------------------------------------------------
# 7. Compute probability-weighted placement distribution and report
# ---------------------------------------------------------------------------

prob_place = np.zeros((N, N), dtype=np.float64)
for r in range(1, N + 1):
    prob_place[:, r - 1] = (
        (rank_matrix == r) * scenario_weights[:, np.newaxis]
    ).sum(axis=0)

expected_rank = (prob_place * np.arange(1, N + 1)).sum(axis=1)
order = np.argsort(expected_rank)

raw_has_place = np.zeros((N, N), dtype=bool)
for r in range(1, N + 1):
    raw_has_place[:, r - 1] = (rank_matrix == r).any(axis=0)

# --- Print results ---
print(f"\nSimulated {8} possible tournament outcomes across {N} brackets.")
print(f"Placements weighted by scenario likelihood  [seed-based, k={SEED_K}].\n")

header = f"{'Bracket':<52} {'CurPts':>6}  {'P(1st)':>7}  {'P(2nd)':>7}  {'Best':>5}  {'ExpRk':>6}"
print(header)
print('-' * len(header))
for b in order:
    name     = bracket_names[b]
    cur      = current_scores[b]
    p_first  = prob_place[b, 0] * 100
    p_second = prob_place[b, 1] * 100
    best     = np.argmax(raw_has_place[b]) + 1
    exp_rk   = expected_rank[b]
    print(f"{name:<52} {cur:>6}  {p_first:>6.2f}%  {p_second:>6.2f}%  {best:>5}  {exp_rk:>6.2f}")

# --- Write detailed CSV ---
out_path = 'simulation_results.csv'
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header_row = (
        ['Bracket', 'CurrentScore', 'ExpectedRank']
        + [f'P_Place_{r}' for r in range(1, N + 1)]
    )
    writer.writerow(header_row)
    for b in order:
        writer.writerow(
            [bracket_names[b], int(current_scores[b]), round(expected_rank[b], 4)]
            + [round(p, 6) for p in prob_place[b].tolist()]
        )

print(f'\nFull placement breakdown written to {out_path}')
