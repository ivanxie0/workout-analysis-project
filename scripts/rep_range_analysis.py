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

DATA_PATH = '../data/workout_data.csv'
OUTPUT_DIR = Path('outputs/rep_range_analysis')
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
        
# ===== PROGRESSION TRACKING =====

def track_progression(df):
    """Calculate weekly best e1RM per exercise per rep range over time."""

    df['week'] = df['date'].dt.to_period('W')

    weekly = (
        df.groupby(['exercise_title', 'rep_range', 'week'], observed=True)
        ['e1rm']
        .max()
        .reset_index()
        .rename(columns={'e1rm': 'best_e1rm'})
    )

    weekly['week_start'] = weekly['week'].apply(lambda w: w.start_time)

    return weekly

# ===== VISUALIZATION =====

def plot_rep_range_distribution(range_dist, output_dir):
    """Bar chart showing how sets are distributed across rep ranges."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    bars = ax.bar(range(len(range_dist)), range_dist['total_sets'], 
                  color=colors)

    ax.set_xticks(range(len(range_dist)))
    ax.set_xticklabels(range_dist.index, fontsize=10)
    ax.set_ylabel('Total Sets')
    ax.set_title('Training Volume by Rep Range')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 20,
                f'{int(height)}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'rep_range_distribution.png', dpi=150)
    plt.close()
    
def plot_e1rm_progression(weekly, output_dir):
    """
    Line chart showing e1RM progression per rep range for focus exercises.
    """
    for exercise in FOCUS_EXERCISES:
        ex_data = weekly[weekly['exercise_title'] == exercise]
        if ex_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = {'Strength (1-3)': '#e74c3c', 'Str-Hyp (4-6)': '#f39c12',
                  'Hypertrophy (7-12)': '#2ecc71', 'Endurance (13-18)':
                      '#3498db'}

        for rep_range in REP_LABELS:
            range_data = ex_data[ex_data['rep_range'] == rep_range]
            if range_data.empty:
                continue
            ax.plot(range_data['week_start'], range_data['best_e1rm'],
                    marker='o', markersize=3, alpha=0.7, label=rep_range,
                    color=colors[rep_range])

        ax.set_xlabel('Date')
        ax.set_ylabel('Estimated 1RM (lbs)')
        ax.set_title(f'{exercise} — e1RM Progression by Rep Range')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        safe_name = (exercise.lower().replace(' ', '_').replace('(', '')
                     .replace(')', ''))
        plt.savefig(output_dir / f'e1rm_progression_{safe_name}.png', dpi=150)
        plt.close()
        
def plot_range_comparison(prs_by_range, output_dir):
    """
    Grouped bar chart comparing e1RM PRs across rep ranges for focus 
    exercises.
    """
    focus_data = prs_by_range[prs_by_range['exercise_title']
                              .isin(FOCUS_EXERCISES)]
    pivot = focus_data.pivot(index='exercise_title', columns='rep_range', 
                             values='pr_e1rm')
    
    pivot = pivot.reindex(FOCUS_EXERCISES).dropna(how='all')

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    pivot.plot(kind='bar', ax=ax, color=colors[:len(pivot.columns)], 
               width=0.7)

    ax.set_ylabel('Estimated 1RM (lbs)')
    ax.set_title('PR Comparison Across Rep Ranges')
    ax.legend(title='Rep Range', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', 
                       fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'rep_range_pr_comparison.png', dpi=150)
    plt.close()
    
    # ===== MAIN =====
 
def main():
 
    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
    # Load and prepare
    df = load_and_prepare(DATA_PATH)
    print(f"Loaded {len(df)} working sets across {df['exercise_title']
          .nunique()} exercises")
    
    # Add calculated columns
    df = add_rep_range(df)
    df = add_estimated_1rm(df)

    # Filter to exercises with enough data
    set_counts = df.groupby('exercise_title').size()
    valid_exercises = set_counts[set_counts >= MIN_SETS].index
    df = df[df['exercise_title'].isin(valid_exercises)].copy()
    print(f"Filtered to {df['exercise_title'].nunique()} exercises with "
          f"{MIN_SETS}+ sets")
    
    # Run analysis
    range_dist, prs_by_range = analyze_rep_ranges(df)
    print_focus_exercise_prs(prs_by_range)
    
    # Progression tracking
    weekly = track_progression(df)
 
    # Generate visualizations
    plot_rep_range_distribution(range_dist, OUTPUT_DIR)
    plot_e1rm_progression(weekly, OUTPUT_DIR)
    plot_range_comparison(prs_by_range, OUTPUT_DIR)
    
    # What rep range do I train in most?
    most_common = range_dist['total_sets'].idxmax()
    most_pct = range_dist.loc[most_common, 'total_sets'] / \
        range_dist['total_sets'].sum() * 100
    print(f"  Most trained range: {most_common} ({most_pct:.0f}% of all sets)")
 
    # What rep range do I least train in?
    least_common = range_dist['total_sets'].idxmin()
    least_pct = range_dist.loc[least_common, 'total_sets'] / \
        range_dist['total_sets'].sum() * 100
    print(f"  Least trained range: {least_common} ({least_pct:.0f}% of all sets)")
 
    print(f"\n  Visualizations saved to: {OUTPUT_DIR}")
    
if __name__ == '__main__':
    main()