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
# 2. Define the 7 remaining future games
#
# All Sweet 16 games are complete. Elite 8 matchups are fully determined:
#   West  E8:  Arizona (1)  vs Purdue (2)
#   South E8:  Iowa (9)     vs Illinois (3)
#   East  E8:  Duke (1)     vs Connecticut (2)
#   Midwest E8: Michigan (1) vs Tennessee (6)
#
# Games still to play:
#   Game 0:  West Elite 8    — Arizona vs Purdue
#   Game 1:  South Elite 8   — Iowa vs Illinois
#   Game 2:  East Elite 8    — Duke vs Connecticut
#   Game 3:  Midwest Elite 8 — Michigan vs Tennessee
#   Game 4:  Final Four (East/South)   — winner of game 2 vs winner of game 1
#   Game 5:  Final Four (West/Midwest) — winner of game 0 vs winner of game 3
#   Game 6:  Championship              — winner of game 4 vs winner of game 5
#
# Total: 7 games → 2^7 = 128 scenarios
# ---------------------------------------------------------------------------

# All 8 remaining teams, paired by game
# Team IDs: 0=Arizona, 1=Purdue, 2=Iowa, 3=Illinois, 4=Duke, 5=Connecticut,
#           6=Michigan, 7=Tennessee
ELITE8 = [
    (0, "West",    "Round of 8", "Arizona",     "Purdue"),
    (1, "South",   "Round of 8", "Iowa",        "Illinois"),
    (2, "East",    "Round of 8", "Duke",        "Connecticut"),
    (3, "Midwest", "Round of 8", "Michigan",    "Tennessee"),
]

all_teams = []
for _, _, _, t1, t2 in ELITE8:
    all_teams.extend([t1, t2])
team_to_id = {t: i for i, t in enumerate(all_teams)}

# Point values for each of the 7 future games
game_points = np.array(
    [point_values["Round of 8"]]  * 4 +    # games 0–3: Elite 8
    [point_values["Round of 4"]]  * 2 +    # games 4–5: Final Four
    [point_values["National Final"]] * 1,  # game 6:    Championship
    dtype=np.int32,
)  # (7,)

# ---------------------------------------------------------------------------
# 2.5. Win-probability matrix (seed-based)
#
#   Seeds are read from winners.csv (Round of 64 rows).
#   P(A beats B) = seed_B^k / (seed_A^k + seed_B^k),  k = 0.80
# ---------------------------------------------------------------------------

SEED_K = 0.80

seed_map = load_seed_map()

seeds = np.array(
    [seed_map.get(t, 8) for t in all_teams], dtype=np.float64
)
s_pow = seeds ** SEED_K
# prob_matrix[i, j] = P(team i beats team j)
prob_matrix = s_pow[np.newaxis, :] / (s_pow[:, np.newaxis] + s_pow[np.newaxis, :])
np.fill_diagonal(prob_matrix, 0.5)

# Display win probabilities for all Elite 8 matchups
print(f"Win probabilities  [seed-based model, k={SEED_K}]:")
print(f"  {'Region':<8} {'Team 1':<16} vs {'Team 2':<16}  P(T1 wins)")
print(f"  {'-'*57}")
for _, region, _, t1, t2 in ELITE8:
    p = prob_matrix[team_to_id[t1], team_to_id[t2]]
    print(f"  {region:<8} {t1:<16}    {t2:<16}  {p:.1%}")

# ---------------------------------------------------------------------------
# 3. Enumerate all 2^7 = 128 scenarios
#    bits[s, g] = 0 → team-1 wins game g;  1 → team-2 wins
# ---------------------------------------------------------------------------

bits = ((np.arange(128, dtype=np.int32)[:, np.newaxis]) >> np.arange(7)) & 1
# shape: (128, 7)

# Elite 8 winners (games 0–3): team IDs 0–7
e8_t1 = np.array([2 * i     for i in range(4)], dtype=np.int32)
e8_t2 = np.array([2 * i + 1 for i in range(4)], dtype=np.int32)
e8_winners = np.where(bits[:, :4] == 0, e8_t1, e8_t2)  # (128, 4)
# columns: 0=West winner, 1=South winner, 2=East winner, 3=Midwest winner

# Final Four (games 4–5)
ff_east_south  = np.where(bits[:, 4:5] == 0, e8_winners[:, 2:3], e8_winners[:, 1:2])  # (128, 1)
ff_west_midwest = np.where(bits[:, 5:6] == 0, e8_winners[:, 0:1], e8_winners[:, 3:4])  # (128, 1)

# Championship (game 6)
champ_winner = np.where(bits[:, 6:7] == 0, ff_east_south, ff_west_midwest)  # (128, 1)

# All 7 game winners as team IDs  (128, 7)
all_winners = np.concatenate(
    [e8_winners, ff_east_south, ff_west_midwest, champ_winner], axis=1
)

# ---------------------------------------------------------------------------
# 3.5. Scenario weights  (probability of each of the 128 outcomes)
# ---------------------------------------------------------------------------

# Elite 8 (games 0–3): all fixed matchups
e8_p1  = prob_matrix[e8_t1, e8_t2]                                           # (4,)
e8_out = np.where(bits[:, :4] == 0, e8_p1[np.newaxis, :], 1.0 - e8_p1)      # (128, 4)

# Final Four (game 4): East E8 winner vs South E8 winner
ff_es_p1  = prob_matrix[e8_winners[:, 2:3], e8_winners[:, 1:2]]              # (128, 1)
ff_es_out = np.where(bits[:, 4:5] == 0, ff_es_p1, 1.0 - ff_es_p1)           # (128, 1)

# Final Four (game 5): West E8 winner vs Midwest E8 winner
ff_wm_p1  = prob_matrix[e8_winners[:, 0:1], e8_winners[:, 3:4]]              # (128, 1)
ff_wm_out = np.where(bits[:, 5:6] == 0, ff_wm_p1, 1.0 - ff_wm_p1)          # (128, 1)

# Championship (game 6)
champ_p1  = prob_matrix[ff_east_south, ff_west_midwest]                      # (128, 1)
champ_out = np.where(bits[:, 6:7] == 0, champ_p1, 1.0 - champ_p1)           # (128, 1)

all_game_probs = np.concatenate(
    [e8_out, ff_es_out, ff_wm_out, champ_out], axis=1
)  # (128, 7)

scenario_weights = all_game_probs.prod(axis=1)
scenario_weights /= scenario_weights.sum()

# ---------------------------------------------------------------------------
# 4. Parse each bracket's predictions for the 7 future games
# ---------------------------------------------------------------------------

def get_future_predictions(path):
    """Return a length-7 list of team IDs (or -1 if team already eliminated)."""
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

    # Games 0–3: Elite 8 (one prediction per region)
    for _, region, rnd, _, _ in ELITE8:
        team = lookup.get((region, rnd), '')
        preds.append(team_to_id.get(team, -1))

    # Games 4–5: Final Four (East/South winner first, West/Midwest second)
    for team in (ff_preds + ['', ''])[:2]:
        preds.append(team_to_id.get(team, -1))

    # Game 6: Championship
    team = lookup.get(('Championship', 'National Final'), '')
    preds.append(team_to_id.get(team, -1))

    return preds

bracket_preds = np.array(
    [get_future_predictions(p) for p in bracket_files],
    dtype=np.int32,
)  # (N, 7)

# ---------------------------------------------------------------------------
# 5. Score all scenarios × brackets simultaneously
# ---------------------------------------------------------------------------

matches = (
    all_winners[:, np.newaxis, :] == bracket_preds[np.newaxis, :, :]
)  # (128, N, 7)

valid = (bracket_preds != -1)[np.newaxis, :, :]
matches &= valid

future_scores = (matches * game_points).sum(axis=2).astype(np.int32)  # (128, N)
total_scores  = current_scores[np.newaxis, :] + future_scores          # (128, N)

# ---------------------------------------------------------------------------
# 6. Rank brackets in each scenario
# ---------------------------------------------------------------------------

rank_matrix = (
    1 + (total_scores[:, :, np.newaxis] > total_scores[:, np.newaxis, :]).sum(axis=1)
)  # (128, N)

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
print(f"\nSimulated {128} possible tournament outcomes across {N} brackets.")
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
