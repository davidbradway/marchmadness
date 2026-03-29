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
# 2. Define the 5 remaining future games
#
# State after Saturday March 28 Elite Eight:
#   West  E8: Arizona (1) beat Purdue (2)   →  Arizona in Final Four
#   South E8: Illinois (3) beat Iowa (9)    →  Illinois in Final Four
#
# Games still to play (Sunday March 29 + Final Four + Championship):
#   Game 0:  East Elite 8    — Duke (1) vs Connecticut (2)
#   Game 1:  Midwest Elite 8 — Michigan (1) vs Tennessee (6)
#   Game 2:  Final Four (East/South)   — winner of game 0 vs Illinois
#   Game 3:  Final Four (West/Midwest) — Arizona vs winner of game 1
#   Game 4:  Championship              — winner of game 2 vs winner of game 3
#
# Total: 5 games → 2^5 = 32 scenarios
# ---------------------------------------------------------------------------

# Team IDs 0–3: the two pending E8 matchups
# Team IDs 4–5: Final Four teams already locked in
# 0=Duke, 1=Connecticut, 2=Michigan, 3=Tennessee, 4=Arizona, 5=Illinois
PENDING_E8 = [
    (0, "East",    "Round of 8", "Duke",    "Connecticut"),
    (1, "Midwest", "Round of 8", "Michigan", "Tennessee"),
]

all_teams = []
for _, _, _, t1, t2 in PENDING_E8:
    all_teams.extend([t1, t2])
all_teams += ["Arizona", "Illinois"]   # IDs 4, 5 — already in Final Four
team_to_id = {t: i for i, t in enumerate(all_teams)}

# Point values for the 5 future games
game_points = np.array(
    [point_values["Round of 8"]]     * 2 +   # games 0–1: Elite 8
    [point_values["Round of 4"]]     * 2 +   # games 2–3: Final Four
    [point_values["National Final"]] * 1,    # game 4:    Championship
    dtype=np.int32,
)  # (5,)

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

# Display win probabilities for pending E8 matchups
print(f"Win probabilities  [seed-based model, k={SEED_K}]:")
print(f"  {'Round':<8} {'Team 1':<16} vs {'Team 2':<16}  P(T1 wins)")
print(f"  {'-'*57}")
for _, region, rnd, t1, t2 in PENDING_E8:
    p = prob_matrix[team_to_id[t1], team_to_id[t2]]
    print(f"  E8  {region:<8} {t1:<16}    {t2:<16}  {p:.1%}")

# ---------------------------------------------------------------------------
# 3. Enumerate all 2^5 = 32 scenarios
#    bits[s, g] = 0 → team-1 wins game g;  1 → team-2 wins
# ---------------------------------------------------------------------------

bits = ((np.arange(32, dtype=np.int32)[:, np.newaxis]) >> np.arange(5)) & 1
# shape: (32, 5)

# Elite 8 (games 0–1): pending matchups
e8_t1 = np.array([0, 2], dtype=np.int32)   # Duke, Michigan
e8_t2 = np.array([1, 3], dtype=np.int32)   # Connecticut, Tennessee
e8_winners = np.where(bits[:, :2] == 0, e8_t1, e8_t2)  # (32, 2)
# column 0 = East winner, column 1 = Midwest winner

# Final Four (games 2–3)
arizona  = team_to_id["Arizona"]    # 4
illinois = team_to_id["Illinois"]   # 5

ff_east_south  = np.where(bits[:, 2:3] == 0, e8_winners[:, 0:1], illinois)   # (32, 1)
ff_west_midwest = np.where(bits[:, 3:4] == 0, arizona, e8_winners[:, 1:2])   # (32, 1)

# Championship (game 4)
champ_winner = np.where(bits[:, 4:5] == 0, ff_east_south, ff_west_midwest)   # (32, 1)

# All 5 game winners  (32, 5)
all_winners = np.concatenate(
    [e8_winners, ff_east_south, ff_west_midwest, champ_winner], axis=1
)

# ---------------------------------------------------------------------------
# 3.5. Scenario weights
# ---------------------------------------------------------------------------

# Elite 8
e8_p1  = prob_matrix[e8_t1, e8_t2]                                            # (2,)
e8_out = np.where(bits[:, :2] == 0, e8_p1[np.newaxis, :], 1.0 - e8_p1)       # (32, 2)

# Final Four (game 2): East E8 winner vs Illinois
ff_es_p1  = prob_matrix[e8_winners[:, 0:1], illinois]                         # (32, 1)
ff_es_out = np.where(bits[:, 2:3] == 0, ff_es_p1, 1.0 - ff_es_p1)            # (32, 1)

# Final Four (game 3): Arizona vs Midwest E8 winner
ff_wm_p1  = prob_matrix[arizona, e8_winners[:, 1:2]]                          # (32, 1)
ff_wm_out = np.where(bits[:, 3:4] == 0, ff_wm_p1, 1.0 - ff_wm_p1)           # (32, 1)

# Championship (game 4)
champ_p1  = prob_matrix[ff_east_south, ff_west_midwest]                       # (32, 1)
champ_out = np.where(bits[:, 4:5] == 0, champ_p1, 1.0 - champ_p1)            # (32, 1)

all_game_probs = np.concatenate(
    [e8_out, ff_es_out, ff_wm_out, champ_out], axis=1
)  # (32, 5)

scenario_weights = all_game_probs.prod(axis=1)
scenario_weights /= scenario_weights.sum()

# ---------------------------------------------------------------------------
# 4. Parse each bracket's predictions for the 5 future games
# ---------------------------------------------------------------------------

def get_future_predictions(path):
    """Return a length-5 list of team IDs (or -1 if team already eliminated)."""
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

    # Games 0–1: pending Elite 8
    for _, region, rnd, _, _ in PENDING_E8:
        team = lookup.get((region, rnd), '')
        preds.append(team_to_id.get(team, -1))

    # Games 2–3: Final Four (East/South winner first, West/Midwest second)
    for team in (ff_preds + ['', ''])[:2]:
        preds.append(team_to_id.get(team, -1))

    # Game 4: Championship
    team = lookup.get(('Championship', 'National Final'), '')
    preds.append(team_to_id.get(team, -1))

    return preds

bracket_preds = np.array(
    [get_future_predictions(p) for p in bracket_files],
    dtype=np.int32,
)  # (N, 5)

# ---------------------------------------------------------------------------
# 5. Score all scenarios × brackets simultaneously
# ---------------------------------------------------------------------------

matches = (
    all_winners[:, np.newaxis, :] == bracket_preds[np.newaxis, :, :]
)  # (32, N, 5)

valid = (bracket_preds != -1)[np.newaxis, :, :]
matches &= valid

future_scores = (matches * game_points).sum(axis=2).astype(np.int32)   # (32, N)
total_scores  = current_scores[np.newaxis, :] + future_scores           # (32, N)

# ---------------------------------------------------------------------------
# 6. Rank brackets in each scenario
# ---------------------------------------------------------------------------

rank_matrix = (
    1 + (total_scores[:, :, np.newaxis] > total_scores[:, np.newaxis, :]).sum(axis=1)
)  # (32, N)

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
print(f"\nSimulated {32} possible tournament outcomes across {N} brackets.")
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
