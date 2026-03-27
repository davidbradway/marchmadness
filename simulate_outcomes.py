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
# 2. Define the 11 remaining future games
#
# State after Thursday March 26 Sweet 16 (West + South complete):
#   West:  Arizona (1) beat Arkansas (4)  →  West E8: Arizona vs Purdue
#          Purdue (2) beat Texas (11)
#   South: Iowa (9) beat Nebraska (4)     →  South E8: Iowa vs Illinois
#          Illinois (3) beat Houston (2)
#
# Games still to play:
#   Games 0–1:  East Sweet 16  — Duke (1) vs St. John's (5)
#                                Michigan St. (3) vs Connecticut (2)
#   Games 2–3:  Midwest Sweet 16 — Michigan (1) vs Alabama (4)
#                                  Tennessee (6) vs Iowa St. (2)
#   Game  4:    West Elite 8   — Arizona (1) vs Purdue (2)   [teams fixed]
#   Game  5:    South Elite 8  — Iowa (9) vs Illinois (3)    [teams fixed]
#   Game  6:    East Elite 8   — winner of game 0 vs winner of game 1
#   Game  7:    Midwest Elite 8 — winner of game 2 vs winner of game 3
#   Game  8:    Final Four (East/South)    — East E8 winner vs South E8 winner
#   Game  9:    Final Four (West/Midwest)  — West E8 winner vs Midwest E8 winner
#   Game 10:    Championship
#
# Total: 11 games → 2^11 = 2,048 scenarios
# ---------------------------------------------------------------------------

EAST_S16 = [
    (0, "East",    "Round of 16", "Duke",        "St. John's"),
    (1, "East",    "Round of 16", "Michigan St.", "Connecticut"),
]
MIDWEST_S16 = [
    (2, "Midwest", "Round of 16", "Michigan",    "Alabama"),
    (3, "Midwest", "Round of 16", "Tennessee",   "Iowa St."),
]
FUTURE_S16 = EAST_S16 + MIDWEST_S16  # games 0–3

# Team IDs 0–7: the 8 remaining Sweet 16 teams (pairs match game index)
# Team IDs 8–9: West E8 fixed teams (Arizona, Purdue)
# Team IDs 10–11: South E8 fixed teams (Iowa, Illinois)
all_teams = []
for _, _, _, t1, t2 in FUTURE_S16:
    all_teams.extend([t1, t2])
all_teams += ["Arizona", "Purdue", "Iowa", "Illinois"]
team_to_id = {t: i for i, t in enumerate(all_teams)}

# Point values for each of the 11 future games
game_points = np.array(
    [point_values["Round of 16"]] * 4 +    # games 0–3: Sweet 16
    [point_values["Round of 8"]]  * 4 +    # games 4–7: Elite 8
    [point_values["Round of 4"]]  * 2 +    # games 8–9: Final Four
    [point_values["National Final"]] * 1,  # game 10:   Championship
    dtype=np.int32
)  # (11,)

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
# prob_matrix[i, j] = P(team i beats team j) = seed_j^k / (seed_i^k + seed_j^k)
prob_matrix = s_pow[np.newaxis, :] / (s_pow[:, np.newaxis] + s_pow[np.newaxis, :])
np.fill_diagonal(prob_matrix, 0.5)

# Display win probabilities for the upcoming Sweet 16 and known E8 matchups
print(f"Win probabilities  [seed-based model, k={SEED_K}]:")
print(f"  {'Round':<8} {'Team 1':<16} vs {'Team 2':<16}  P(T1 wins)")
print(f"  {'-'*57}")
for _, _, _, t1, t2 in FUTURE_S16:
    p = prob_matrix[team_to_id[t1], team_to_id[t2]]
    print(f"  S16      {t1:<16}    {t2:<16}  {p:.1%}")
for t1, t2 in [("Arizona", "Purdue"), ("Iowa", "Illinois")]:
    p = prob_matrix[team_to_id[t1], team_to_id[t2]]
    print(f"  E8       {t1:<16}    {t2:<16}  {p:.1%}")

# ---------------------------------------------------------------------------
# 3. Enumerate all 2^11 = 2,048 scenarios
#    bits[s, g] = 0 → team-1 side wins game g;  1 → team-2 side wins
# ---------------------------------------------------------------------------

bits = ((np.arange(2048, dtype=np.int32)[:, np.newaxis]) >> np.arange(11)) & 1
# shape: (2048, 11)

# Sweet 16 winners (games 0–3): team IDs 0–7
s16_t1 = np.array([2 * i     for i in range(4)], dtype=np.int32)
s16_t2 = np.array([2 * i + 1 for i in range(4)], dtype=np.int32)
s16_winners = np.where(bits[:, :4] == 0, s16_t1, s16_t2)  # (2048, 4)

# West E8 (game 4): Arizona (8) vs Purdue (9) — fixed teams
west_e8_t1 = team_to_id["Arizona"]   # 8
west_e8_t2 = team_to_id["Purdue"]    # 9
west_e8_winner = np.where(bits[:, 4:5] == 0, west_e8_t1, west_e8_t2)  # (2048, 1)

# South E8 (game 5): Iowa (10) vs Illinois (11) — fixed teams
south_e8_t1 = team_to_id["Iowa"]     # 10
south_e8_t2 = team_to_id["Illinois"] # 11
south_e8_winner = np.where(bits[:, 5:6] == 0, south_e8_t1, south_e8_t2)  # (2048, 1)

# East E8 (game 6): East S16 game 0 winner vs East S16 game 1 winner
east_e8_winner = np.where(
    bits[:, 6:7] == 0, s16_winners[:, 0:1], s16_winners[:, 1:2]
)  # (2048, 1)

# Midwest E8 (game 7): Midwest S16 game 0 winner vs Midwest S16 game 1 winner
midwest_e8_winner = np.where(
    bits[:, 7:8] == 0, s16_winners[:, 2:3], s16_winners[:, 3:4]
)  # (2048, 1)

# Final Four (games 8–9)
ff_east_south = np.where(
    bits[:, 8:9] == 0, east_e8_winner, south_e8_winner
)  # (2048, 1)
ff_west_midwest = np.where(
    bits[:, 9:10] == 0, west_e8_winner, midwest_e8_winner
)  # (2048, 1)

# Championship (game 10)
champ_winner = np.where(
    bits[:, 10:11] == 0, ff_east_south, ff_west_midwest
)  # (2048, 1)

# All 11 game winners as team IDs  (2048, 11)
all_winners = np.concatenate(
    [s16_winners, west_e8_winner, south_e8_winner,
     east_e8_winner, midwest_e8_winner,
     ff_east_south, ff_west_midwest, champ_winner],
    axis=1
)

# ---------------------------------------------------------------------------
# 3.5. Scenario weights  (probability of each of the 2,048 outcomes)
# ---------------------------------------------------------------------------

# Sweet 16 (games 0–3): fixed matchups
s16_p1 = prob_matrix[s16_t1, s16_t2]                                          # (4,)
s16_out = np.where(bits[:, :4] == 0, s16_p1[np.newaxis, :], 1.0 - s16_p1)    # (2048, 4)

# West E8 (game 4): fixed teams
west_e8_p1 = prob_matrix[west_e8_t1, west_e8_t2]                              # scalar
west_e8_out = np.where(bits[:, 4:5] == 0, west_e8_p1, 1.0 - west_e8_p1)      # (2048, 1)

# South E8 (game 5): fixed teams
south_e8_p1 = prob_matrix[south_e8_t1, south_e8_t2]                           # scalar
south_e8_out = np.where(bits[:, 5:6] == 0, south_e8_p1, 1.0 - south_e8_p1)   # (2048, 1)

# East E8 (game 6): teams depend on S16
east_e8_p1 = prob_matrix[s16_winners[:, 0:1], s16_winners[:, 1:2]]            # (2048, 1)
east_e8_out = np.where(bits[:, 6:7] == 0, east_e8_p1, 1.0 - east_e8_p1)      # (2048, 1)

# Midwest E8 (game 7): teams depend on S16
midwest_e8_p1 = prob_matrix[s16_winners[:, 2:3], s16_winners[:, 3:4]]         # (2048, 1)
midwest_e8_out = np.where(bits[:, 7:8] == 0, midwest_e8_p1, 1.0 - midwest_e8_p1)  # (2048, 1)

# Final Four (game 8): East E8 winner vs South E8 winner
ff_es_p1 = prob_matrix[east_e8_winner, south_e8_winner]                       # (2048, 1)
ff_es_out = np.where(bits[:, 8:9] == 0, ff_es_p1, 1.0 - ff_es_p1)            # (2048, 1)

# Final Four (game 9): West E8 winner vs Midwest E8 winner
ff_wm_p1 = prob_matrix[west_e8_winner, midwest_e8_winner]                     # (2048, 1)
ff_wm_out = np.where(bits[:, 9:10] == 0, ff_wm_p1, 1.0 - ff_wm_p1)          # (2048, 1)

# Championship (game 10)
champ_p1 = prob_matrix[ff_east_south, ff_west_midwest]                        # (2048, 1)
champ_out = np.where(bits[:, 10:11] == 0, champ_p1, 1.0 - champ_p1)          # (2048, 1)

all_game_probs = np.concatenate(
    [s16_out, west_e8_out, south_e8_out, east_e8_out, midwest_e8_out,
     ff_es_out, ff_wm_out, champ_out],
    axis=1
)  # (2048, 11)

scenario_weights = all_game_probs.prod(axis=1)
scenario_weights /= scenario_weights.sum()

# ---------------------------------------------------------------------------
# 4. Parse each bracket's predictions for the 11 future games
# ---------------------------------------------------------------------------

def get_future_predictions(path):
    """Return a length-11 list of team IDs (or -1 if team already eliminated)."""
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

    # Games 0–3: East and Midwest Sweet 16
    for _, region, rnd, _, _ in FUTURE_S16:
        team = lookup.get((region, rnd), '')
        preds.append(team_to_id.get(team, -1))

    # Game 4: West Elite 8 (Arizona vs Purdue)
    team = lookup.get(('West', 'Round of 8'), '')
    preds.append(team_to_id.get(team, -1))

    # Game 5: South Elite 8 (Iowa vs Illinois)
    team = lookup.get(('South', 'Round of 8'), '')
    preds.append(team_to_id.get(team, -1))

    # Game 6: East Elite 8
    team = lookup.get(('East', 'Round of 8'), '')
    preds.append(team_to_id.get(team, -1))

    # Game 7: Midwest Elite 8
    team = lookup.get(('Midwest', 'Round of 8'), '')
    preds.append(team_to_id.get(team, -1))

    # Games 8–9: Final Four (East/South winner first, West/Midwest second)
    for team in (ff_preds + ['', ''])[:2]:
        preds.append(team_to_id.get(team, -1))

    # Game 10: Championship
    team = lookup.get(('Championship', 'National Final'), '')
    preds.append(team_to_id.get(team, -1))

    return preds

bracket_preds = np.array(
    [get_future_predictions(p) for p in bracket_files],
    dtype=np.int32
)  # (N, 11)

# ---------------------------------------------------------------------------
# 5. Score all scenarios × brackets simultaneously
# ---------------------------------------------------------------------------

matches = (
    all_winners[:, np.newaxis, :] == bracket_preds[np.newaxis, :, :]
)  # (2048, N, 11)

valid = (bracket_preds != -1)[np.newaxis, :, :]
matches &= valid

future_scores = (matches * game_points).sum(axis=2).astype(np.int32)  # (2048, N)
total_scores  = current_scores[np.newaxis, :] + future_scores          # (2048, N)

# ---------------------------------------------------------------------------
# 6. Rank brackets in each scenario
# ---------------------------------------------------------------------------

rank_matrix = (
    1 + (total_scores[:, :, np.newaxis] > total_scores[:, np.newaxis, :]).sum(axis=1)
)  # (2048, N)

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
print(f"\nSimulated {2048} possible tournament outcomes across {N} brackets.")
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
