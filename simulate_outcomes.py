import csv
import glob
import os
import numpy as np

# ---------------------------------------------------------------------------
# 1. Load current (already-scored) state
# ---------------------------------------------------------------------------

point_values = {}
with open('group_scoring.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        rnd = row['Round'].strip()
        pts = row['Correct Picks'].strip().replace(' pts', '').replace(' pt', '')
        try:
            point_values[rnd] = int(pts)
        except ValueError:
            pass

actual_winners = {}
with open('winners.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        actual_winners[(row['Round'].strip(), row['Winner'].strip())] = True

bracket_files = sorted(glob.glob('brackets/*.csv'))
bracket_names = [os.path.basename(p) for p in bracket_files]
N = len(bracket_files)

current_scores = []
for path in bracket_files:
    score = 0
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['Round'].strip(), row['Predicted Winner'].strip())
            if key in actual_winners:
                score += point_values.get(row['Round'].strip(), 0)
    current_scores.append(score)

current_scores = np.array(current_scores, dtype=np.int32)  # (N,)

# ---------------------------------------------------------------------------
# 2. Define the 15 future games
#
# Games 0–7:  Sweet 16  (Round of 16) — teams are known
# Games 8–11: Elite 8   (Round of 8)  — teams = winners of games {2k, 2k+1}
# Games 12–13: Final Four (Round of 4) — East/South winner vs West/Midwest winner
# Game 14:    Championship (National Final)
#
# Sweet 16 matchups derived from Round-of-32 results in winners.csv:
#   East:    Duke vs St. John's  |  Michigan St. vs Connecticut
#   South:   Iowa vs Nebraska    |  Illinois vs Houston
#   West:    Arizona vs Arkansas |  Texas vs Purdue
#   Midwest: Michigan vs Alabama |  Tennessee vs Iowa St.
# ---------------------------------------------------------------------------

# (game_index, region, round_name, team1, team2)
SWEET16 = [
    (0, "East",    "Round of 16", "Duke",        "St. John's"),
    (1, "East",    "Round of 16", "Michigan St.", "Connecticut"),
    (2, "South",   "Round of 16", "Iowa",         "Nebraska"),
    (3, "South",   "Round of 16", "Illinois",     "Houston"),
    (4, "West",    "Round of 16", "Arizona",      "Arkansas"),
    (5, "West",    "Round of 16", "Texas",        "Purdue"),
    (6, "Midwest", "Round of 16", "Michigan",     "Alabama"),
    (7, "Midwest", "Round of 16", "Tennessee",    "Iowa St."),
]
E8_REGIONS = ["East", "South", "West", "Midwest"]

# Build team-ID lookup: team_to_id[name] -> 0..15
# IDs 2i and 2i+1 are the two teams of Sweet 16 game i
all_s16_teams = []
for _, _, _, t1, t2 in SWEET16:
    all_s16_teams.extend([t1, t2])
team_to_id = {t: i for i, t in enumerate(all_s16_teams)}

# Point values for each of the 15 games, in order
game_points = np.array(
    [point_values["Round of 16"]] * 8 +
    [point_values["Round of 8"]]  * 4 +
    [point_values["Round of 4"]]  * 2 +
    [point_values["National Final"]] * 1,
    dtype=np.int32
)  # (15,)

# ---------------------------------------------------------------------------
# 2.5. Team strengths and win-probability matrix
#
#  Two modes (priority order):
#
#  1. team_strengths.csv  (columns: Team, Strength)
#     "Strength" can be any rating where higher = better, e.g. KenPom AdjEM.
#     Win prob uses a logistic model:
#       P(A beats B) = 1 / (1 + exp(-(str_A - str_B) / STRENGTH_SCALE))
#     STRENGTH_SCALE = 10 is calibrated for KenPom AdjEM units (a 10-pt AdjEM
#     edge corresponds to ~75% win probability on a neutral court).
#
#     To create this file, visit barttorvik.com or kenpom.com and add a row per
#     team, e.g.:
#       Team,Strength
#       Duke,32.1
#       Connecticut,27.4
#       ...
#
#  2. Seed-based fallback (no external file needed)
#     Seeds are read automatically from winners.csv (Round of 64 rows).
#     Win prob:  P(A beats B) = seed_B^k / (seed_A^k + seed_B^k)
#     k = SEED_K = 0.80  (calibrated to NCAA tournament historical data)
#     Example: Duke (1) vs St. John's (5) → P(Duke) ≈ 78%
# ---------------------------------------------------------------------------

SEED_K        = 0.80   # seed-model exponent; raise to penalise upsets more
STRENGTH_SCALE = 10.0  # logistic scale for custom-strength model

# Build seed map from Round of 64 data (all 64 original teams)
seed_map = {}
with open('winners.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['Round'].strip() == 'Round of 64':
            try:
                seed_map[row['Team 1'].strip()] = int(row['Team 1 Seed'].strip())
                seed_map[row['Team 2'].strip()] = int(row['Team 2 Seed'].strip())
            except (ValueError, KeyError):
                pass

# Optionally load custom team strengths
strength_map = {}
if os.path.exists('team_strengths.csv'):
    with open('team_strengths.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                strength_map[row['Team'].strip()] = float(row['Strength'].strip())
            except (ValueError, KeyError):
                pass
    print(f"Loaded team_strengths.csv ({len(strength_map)} teams).")

# Build 16×16 win-probability matrix  prob_matrix[i, j] = P(team i beats team j)
if strength_map:
    team_strengths = np.array(
        [strength_map.get(t, 0.0) for t in all_s16_teams], dtype=np.float64
    )
    diff = team_strengths[:, np.newaxis] - team_strengths[np.newaxis, :]  # (16,16)
    prob_matrix = 1.0 / (1.0 + np.exp(-diff / STRENGTH_SCALE))
    strength_source = f"team_strengths.csv  (logistic, scale={STRENGTH_SCALE})"
else:
    seeds = np.array(
        [seed_map.get(t, 8) for t in all_s16_teams], dtype=np.float64
    )
    s_pow = seeds ** SEED_K
    # prob_matrix[i,j] = P(i beats j) = seed_j^k / (seed_i^k + seed_j^k)
    prob_matrix = s_pow[np.newaxis, :] / (s_pow[:, np.newaxis] + s_pow[np.newaxis, :])
    strength_source = f"seed-based model  (k={SEED_K})"

np.fill_diagonal(prob_matrix, 0.5)  # self-play is irrelevant; set to 0.5

# Display win probabilities for the known Sweet 16 matchups
print(f"\nWin probabilities  [{strength_source}]:")
print(f"  {'Game':<5} {'Team 1':<16} vs {'Team 2':<16}  P(T1 wins)")
print(f"  {'-'*57}")
for i, (_, region, _, t1, t2) in enumerate(SWEET16):
    p = prob_matrix[team_to_id[t1], team_to_id[t2]]
    print(f"  S16   {t1:<16}    {t2:<16}  {p:.1%}")

# ---------------------------------------------------------------------------
# 3. Enumerate all 2^15 = 32,768 scenarios
#    bits[s, g] = 0 means "left/team-1 side wins game g", 1 = right/team-2 side
# ---------------------------------------------------------------------------

bits = ((np.arange(32768, dtype=np.int32)[:, np.newaxis]) >> np.arange(15)) & 1
# shape: (32768, 15)

# S16 winners as team IDs
s16_t1 = np.array([2 * i     for i in range(8)], dtype=np.int32)
s16_t2 = np.array([2 * i + 1 for i in range(8)], dtype=np.int32)
s16_winners = np.where(bits[:, :8] == 0, s16_t1, s16_t2)  # (32768, 8)

# E8: game 8+k pairs S16 games 2k and 2k+1
e8_winners = np.where(
    bits[:, 8:12] == 0,
    s16_winners[:, [0, 2, 4, 6]],   # left S16 winner
    s16_winners[:, [1, 3, 5, 7]],   # right S16 winner
)  # (32768, 4)

# Final Four: game 12 = East vs South E8 winners; game 13 = West vs Midwest
ff_winners = np.where(
    bits[:, 12:14] == 0,
    e8_winners[:, [0, 2]],
    e8_winners[:, [1, 3]],
)  # (32768, 2)

# Championship: game 14
champ_winner = np.where(
    bits[:, 14:15] == 0,
    ff_winners[:, :1],
    ff_winners[:, 1:],
)  # (32768, 1)

# All 15 game winners as team IDs
all_winners = np.concatenate(
    [s16_winners, e8_winners, ff_winners, champ_winner], axis=1
)  # (32768, 15)

# ---------------------------------------------------------------------------
# 3.5. Scenario weights  (probability that each specific sequence of outcomes occurs)
#
#   For every game in every scenario look up P(the winner of that game wins) from
#   prob_matrix, then multiply all 15 per-game probabilities to get P(scenario).
#
#   Later rounds use the team IDs determined by earlier-round outcomes, so each
#   scenario carries a path-consistent probability through the bracket.
# ---------------------------------------------------------------------------

# Sweet 16: matchups are fixed — one probability per game, broadcast across scenarios
s16_p1 = prob_matrix[s16_t1, s16_t2]                                         # (8,)
s16_out = np.where(bits[:, :8] == 0, s16_p1[np.newaxis, :], 1.0 - s16_p1)   # (32768, 8)

# Elite 8: team IDs depend on S16 outcomes
e8_t1_ids = s16_winners[:, [0, 2, 4, 6]]        # (32768, 4)  left S16 winners
e8_t2_ids = s16_winners[:, [1, 3, 5, 7]]        # (32768, 4)  right S16 winners
e8_p1  = prob_matrix[e8_t1_ids, e8_t2_ids]      # (32768, 4)
e8_out = np.where(bits[:, 8:12] == 0, e8_p1, 1.0 - e8_p1)    # (32768, 4)

# Final Four: team IDs depend on E8 outcomes
ff_t1_ids = e8_winners[:, [0, 2]]               # (32768, 2)  E8 winners: East, West
ff_t2_ids = e8_winners[:, [1, 3]]               # (32768, 2)  E8 winners: South, Midwest
ff_p1  = prob_matrix[ff_t1_ids, ff_t2_ids]      # (32768, 2)
ff_out = np.where(bits[:, 12:14] == 0, ff_p1, 1.0 - ff_p1)   # (32768, 2)

# Championship: team IDs depend on FF outcomes
champ_t1_ids = ff_winners[:, :1]                # (32768, 1)
champ_t2_ids = ff_winners[:, 1:]                # (32768, 1)
champ_p1  = prob_matrix[champ_t1_ids, champ_t2_ids]           # (32768, 1)
champ_out = np.where(bits[:, 14:15] == 0, champ_p1, 1.0 - champ_p1)  # (32768, 1)

# Product of all 15 per-game probabilities → unnormalized scenario weight
all_game_probs = np.concatenate(
    [s16_out, e8_out, ff_out, champ_out], axis=1
)  # (32768, 15)
scenario_weights = all_game_probs.prod(axis=1)   # (32768,)
scenario_weights /= scenario_weights.sum()        # normalise to sum = 1

# ---------------------------------------------------------------------------
# 4. Parse each bracket's predictions for the 15 future games
# ---------------------------------------------------------------------------

def get_future_predictions(path):
    """Return a length-15 list of team IDs (or -1 if team not in Sweet 16)."""
    # Read all rows once
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Build lookup: (region, round) -> predicted winner
    lookup = {}
    ff_preds = []
    for row in rows:
        region = row['Region'].strip()
        rnd    = row['Round'].strip()
        winner = row['Predicted Winner'].strip()
        if region == 'Final Four' and rnd == 'Round of 4':
            ff_preds.append(winner)  # preserve order (East/South first, West/Midwest second)
        else:
            lookup[(region, rnd)] = winner

    preds = []

    # Games 0–7: Sweet 16
    for _, region, rnd, _, _ in SWEET16:
        team = lookup.get((region, rnd), '')
        preds.append(team_to_id.get(team, -1))

    # Games 8–11: Elite 8
    for region in E8_REGIONS:
        team = lookup.get((region, 'Round of 8'), '')
        preds.append(team_to_id.get(team, -1))

    # Games 12–13: Final Four (two rows, in order)
    for team in (ff_preds + ['', ''])[:2]:
        preds.append(team_to_id.get(team, -1))

    # Game 14: Championship
    team = lookup.get(('Championship', 'National Final'), '')
    preds.append(team_to_id.get(team, -1))

    return preds

bracket_preds = np.array(
    [get_future_predictions(p) for p in bracket_files],
    dtype=np.int32
)  # (N, 15)

# ---------------------------------------------------------------------------
# 5. Score all scenarios × brackets simultaneously
# ---------------------------------------------------------------------------

# matches[s, b, g] = True if bracket b's game-g prediction == actual winner in scenario s
matches = (
    all_winners[:, np.newaxis, :] == bracket_preds[np.newaxis, :, :]
)  # (32768, N, 15)

# Ignore predictions of -1 (team eliminated before Sweet 16 — can never score)
valid = (bracket_preds != -1)[np.newaxis, :, :]  # (1, N, 15)
matches &= valid

future_scores = (matches * game_points).sum(axis=2).astype(np.int32)  # (32768, N)
total_scores  = current_scores[np.newaxis, :] + future_scores          # (32768, N)

# ---------------------------------------------------------------------------
# 6. Rank brackets in each scenario
#    rank[s, b] = 1 + number of brackets with a strictly higher score than b
#    Ties share the same (lowest) rank.
# ---------------------------------------------------------------------------

rank_matrix = (
    1 + (total_scores[:, :, np.newaxis] > total_scores[:, np.newaxis, :]).sum(axis=1)
)  # (32768, N)

# ---------------------------------------------------------------------------
# 7. Compute probability-weighted placement distribution and report
#
#   prob_place[b, r] = probability (weighted by scenario likelihood) that
#                      bracket b finishes in place r+1.
#   expected_rank[b] = sum_r  r * prob_place[b, r-1]   (probability-weighted)
# ---------------------------------------------------------------------------

prob_place = np.zeros((N, N), dtype=np.float64)
for r in range(1, N + 1):
    # weight each scenario by its likelihood before counting
    prob_place[:, r - 1] = (
        (rank_matrix == r) * scenario_weights[:, np.newaxis]
    ).sum(axis=0)

# Expected rank (probability-weighted average)
expected_rank = (prob_place * np.arange(1, N + 1)).sum(axis=1)
order = np.argsort(expected_rank)

# Best-case finish: smallest rank achievable in at least one scenario
raw_has_place = np.zeros((N, N), dtype=bool)
for r in range(1, N + 1):
    raw_has_place[:, r - 1] = (rank_matrix == r).any(axis=0)

# --- Print results ---
print(f"\nSimulated {32768} possible tournament outcomes across {N} brackets.")
print(f"Placements weighted by scenario likelihood  [{strength_source}].\n")

header = f"{'Bracket':<52} {'CurPts':>6}  {'P(1st)':>7}  {'P(2nd)':>7}  {'Best':>5}  {'ExpRk':>6}"
print(header)
print('-' * len(header))
for b in order:
    name     = bracket_names[b]
    cur      = current_scores[b]
    p_first  = prob_place[b, 0] * 100   # as percentage
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
