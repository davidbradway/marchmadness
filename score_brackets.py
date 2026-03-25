import csv
import os
import glob

# Load point values from group_scoring.csv
point_values = {}
with open('group_scoring.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        round_name = row['Round'].strip()
        pts_str = row['Correct Picks'].strip().replace(' pts', '').replace(' pt', '')
        try:
            point_values[round_name] = int(pts_str)
        except ValueError:
            pass

# Load actual winners keyed by (Round, Winner)
actual_winners = {}
with open('winners.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['Round'].strip(), row['Winner'].strip())
        actual_winners[key] = True # We only care about the winner, not the matchup details

# Score each bracket file
results = []
for path in sorted(glob.glob('bracket_*.csv')):
    filename = os.path.basename(path)
    score = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['Round'].strip(), row['Predicted Winner'].strip())
            if key in actual_winners:
                round_name = row['Round'].strip()
                score += point_values.get(round_name, 0)
    results.append((filename, score))

# Sort by score descending
results.sort(key=lambda x: x[1], reverse=True)

# Write standings.csv
with open('standings.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Rank', 'Bracket', 'Score'])
    for rank, (filename, score) in enumerate(results, start=1):
        writer.writerow([rank, filename, score])

print(f"standings.csv written with {len(results)} brackets.")
for rank, (filename, score) in enumerate(results, start=1):
    print(f"{rank:2}. {score:4} pts  {filename}")
