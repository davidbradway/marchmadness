import csv
import glob
import os

from tournament import load_point_values, load_actual_winners, load_all_scores

point_values   = load_point_values()
actual_winners = load_actual_winners()

bracket_files = sorted(glob.glob('brackets/*.csv'))
scores = load_all_scores(bracket_files, actual_winners, point_values)

results = sorted(
    zip([os.path.basename(p) for p in bracket_files], scores),
    key=lambda x: x[1],
    reverse=True,
)

# Write standings.csv
with open('standings.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Rank', 'Bracket', 'Score'])
    for rank, (filename, score) in enumerate(results, start=1):
        writer.writerow([rank, filename, score])

print(f"standings.csv written with {len(results)} brackets.")
for rank, (filename, score) in enumerate(results, start=1):
    print(f"{rank:2}. {score:4} pts  {filename}")
