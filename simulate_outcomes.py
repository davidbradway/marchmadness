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
# 7. Count placements and report
# ---------------------------------------------------------------------------

# placement_counts[b, r] = # scenarios where bracket b finishes in place r+1
placement_counts = np.zeros((N, N), dtype=np.int32)
for r in range(1, N + 1):
    placement_counts[:, r - 1] = (rank_matrix == r).sum(axis=0)

# Sort output by expected rank (weighted average rank across all scenarios)
expected_rank = (placement_counts * np.arange(1, N + 1)).sum(axis=1) / 32768
order = np.argsort(expected_rank)

# --- Print results ---
print(f"Simulated {32768} possible tournament outcomes across {N} brackets.\n")

# Summary: Current points / 1st-place count / 2nd-place count / best-case / most-likely finish
header = f"{'Bracket':<52} {'CurPts':>6}  {'1st':>6}  {'2nd':>6}  {'Best':>5}  {'ExpRk':>6}"
print(header)
print('-' * len(header))
for b in order:
    name        = bracket_names[b]
    cur         = current_scores[b]
    firsts      = placement_counts[b, 0]
    seconds     = placement_counts[b, 1]
    best        = np.argmax(placement_counts[b] > 0) + 1  # best possible finish
    exp_rk      = expected_rank[b]
    print(f"{name:<52} {cur:>6}  {firsts:>6}  {seconds:>6}  {best:>5}  {exp_rk:>6.2f}")

# --- Write detailed CSV ---
out_path = 'simulation_results.csv'
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header_row = ['Bracket', 'CurrentScore', 'ExpectedRank'] + [f'Place_{r}' for r in range(1, N + 1)]
    writer.writerow(header_row)
    for b in order:
        writer.writerow(
            [bracket_names[b], int(current_scores[b]), round(expected_rank[b], 3)]
            + placement_counts[b].tolist()
        )

print(f'\nFull placement breakdown written to {out_path}')
