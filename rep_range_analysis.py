"""
Rep Range Analyzer
==================
Categorize training sets by rep range, calculate estimated 1RM,
and track strength progression across different training zones.

"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CONFIGURATION =====

DATA_PATH = 'workout_data.csv'
OUTPUT_DIR = Path('/outputs/rep_range_analysis')
MIN_SETS = 20
MAX_REPS = 18

REP_BINS = [0, 3, 6, 12, MAX_REPS]
REP_LABELS = ['Strength (1-3)', 'Str-Hyp (4-6)', 'Hypertrophy (7-12)', 
              'Endurance (13-18)']

FOCUS_EXERCISES = [
    'Dumbbell Press (Combined)',
    'Lat Pulldown (Cable)',
    'Leg Extension (Machine)',
    'Triceps Extension (Cable)',
    'Lateral Raise (Cable)',
]

EXERCISE_MERGE_MAP = {
    'Bench Press (Dumbbell)': 'Dumbbell Press (Combined)',
    'Incline Bench Press (Dumbbell)': 'Dumbbell Press (Combined)',
}

def load_and_prepare(filepath):
    """Load workout data and add calculated columns."""
    df = pd.read_csv(filepath)
    
    df['date'] = pd.to_datetime(df['start_time'], format='%d %b %Y, %H:%M')
    df = df.dropna(subset=['weight_lbs', 'reps'])
    df = df[df['set_type'] != 'warmup'].copy()
    df = df[df['reps'] <= MAX_REPS]

    df['exercise_title'] = df['exercise_title'].replace(EXERCISE_MERGE_MAP)

    return df

def add_rep_range(df):
    """Categorize each set into a rep range"""
    df['rep_range'] = pd.cut(
        df['reps'],
        bins=REP_BINS,
        labels=REP_LABELS,
        right=True
    )
    return df

def add_estimated_1rm(df):
    """Calculate estimated 1RM for each set using the Epley formula."""
    df['e1rm'] = df['weight_lbs'] * (1 + df['reps'] / 30)
    return df

def analyze_rep_ranges(df):
    """Analyze rep range distribution and PRs per range."""
    range_dist = df.groupby('rep_range', observed=True).agg(
        total_sets=('set_index', 'count'),
        avg_weight=('weight_lbs', 'mean'),
        avg_e1rm=('e1rm', 'mean')
    ).round(1)

    print("=" * 60)
    print("REP RANGE DISTRIBUTION (ALL EXERCISES)")
    print("=" * 60)
    for range_name, row in range_dist.iterrows():
        pct = row['total_sets'] / len(df) * 100
        print(f"  {range_name:<22} {int(row['total_sets']):>5} sets "
              f"({pct:>5.1f}%)  "
              f"avg weight: {row['avg_weight']:>6.1f} lbs  avg e1RM: "
              f"{row['avg_e1rm']:>6.1f} lbs")
        
    prs_by_range = (
        df.groupby(['exercise_title', 'rep_range'], observed=True)['e1rm']
        .max()
        .reset_index()
        .rename(columns={'e1rm': 'pr_e1rm'})
    )

    return range_dist, prs_by_range

def print_focus_exercise_prs(prs_by_range):
    """Print PR table for focus exercises, broken down by rep range."""

    print("\n" + "=" * 60)
    print("ESTIMATED 1RM PRs BY REP RANGE (FOCUS EXERCISES)")
    print("=" * 60)

    for exercise in FOCUS_EXERCISES:
        ex_data = prs_by_range[prs_by_range['exercise_title'] == exercise]
        if ex_data.empty:
            continue

        print(f"\n  {exercise}")
        print(f"  {'-' * 50}")

        for _, row in ex_data.iterrows():
            print(f"    {row['rep_range']:<22} e1RM: {row['pr_e1rm']:>7.1f} "
                  f"lbs")

        best = ex_data['pr_e1rm'].max()
        worst = ex_data['pr_e1rm'].min()
        gap_pct = (best - worst) / best * 100
        print(f"    {'Range gap:':<22}        {gap_pct:>5.1f}%")


