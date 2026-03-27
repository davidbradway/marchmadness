"""Shared data-loading helpers used by score_brackets.py and simulate_outcomes.py."""

import csv
import glob
import os


def load_point_values(path='group_scoring.csv'):
    """Return {round_name: points} from group_scoring.csv."""
    point_values = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rnd = row['Round'].strip()
            pts = row['Correct Picks'].strip().replace(' pts', '').replace(' pt', '')
            try:
                point_values[rnd] = int(pts)
            except ValueError:
                pass
    return point_values


def load_actual_winners(path='winners.csv'):
    """Return {(round, winner): True} from winners.csv."""
    actual_winners = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            actual_winners[(row['Round'].strip(), row['Winner'].strip())] = True
    return actual_winners


def load_seed_map(path='winners.csv'):
    """Return {team_name: seed} for all 64 teams from the Round of 64 rows."""
    seed_map = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['Round'].strip() == 'Round of 64':
                try:
                    seed_map[row['Team 1'].strip()] = int(row['Team 1 Seed'].strip())
                    seed_map[row['Team 2'].strip()] = int(row['Team 2 Seed'].strip())
                except (ValueError, KeyError):
                    pass
    return seed_map


def score_bracket(path, actual_winners, point_values):
    """Return the total score for a single bracket file."""
    score = 0
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['Round'].strip(), row['Predicted Winner'].strip())
            if key in actual_winners:
                score += point_values.get(row['Round'].strip(), 0)
    return score


def load_all_scores(bracket_files, actual_winners, point_values):
    """Return a list of scores in the same order as bracket_files."""
    return [score_bracket(p, actual_winners, point_values) for p in bracket_files]
