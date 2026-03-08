"""
Advanced Volume Analyzer
========================
Corrects for reverse causation: PRs causing volume increases
Separates training volume from PR volume for accurate correlation analysis

Author: Ivan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== CONFIGURATION =====
DATA_PATH = 'workout_data.csv'
BEGINNER_WEEKS = 26
INTERMEDIATE_WEEKS = 52
DATE_FORMAT = '%d %b %Y, %H:%M'

# ===== HELPER FUNCTIONS =====

def setup_output_folder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vis_folder = os.path.join(script_dir, 'visualizations')
    os.makedirs(vis_folder, exist_ok=True)
    return vis_folder

def load_workout_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['start_time'], format=DATE_FORMAT)
    df['volume'] = df['reps'] * df['weight_lbs']
    df['week'] = df['date'].dt.to_period('W')
    return df

def identify_pr_sets(df):
    # Compares running max to previous running max per exercise
    # to detect when a new PR weight is hit
    df_sorted = df.sort_values(['exercise_title', 'date'])
    df_sorted['running_max'] = df_sorted.groupby('exercise_title')['weight_lbs'].cummax()
    df_sorted['is_pr'] = (
        df_sorted['running_max'] != 
        df_sorted.groupby('exercise_title')['running_max'].shift()
    )
    df_sorted['week'] = df_sorted['date'].dt.to_period('W')
    return df_sorted

def calculate_weekly_metrics(df):
    # Separates PR sets from training sets to avoid reverse causation
    # (PRs inflate volume, distorting correlation)
    # Calculate both total and training-only volume
    weekly_volume_total = df.groupby('week')['volume'].sum()
    
    non_pr_sets = df[df['is_pr'] == False]
    weekly_volume_training = non_pr_sets.groupby('week')['volume'].sum()

    weekly_sets = df.groupby('week').size()
    weekly_prs = df[df['is_pr']].groupby('week').size()

    # Combine into a summary
    weekly_metrics = pd.DataFrame({
        'volume_total': weekly_volume_total,
        'volume_training': weekly_volume_training,
        'sets': weekly_sets,
        'prs': weekly_prs
    }).fillna(0)

    weekly_metrics['weeks_training'] = range(len(weekly_metrics))

    return weekly_metrics

def add_lagged_features(weekly_df):
    weekly_df['volume_prev_week'] = weekly_df['volume_training'].shift(1)
    weekly_df['sets_prev_week'] = weekly_df['sets'].shift(1)
    return weekly_df

def split_by_training_phase(weekly_df):
    beginner = weekly_df[weekly_df['weeks_training'] <= BEGINNER_WEEKS]
    intermediate = weekly_df[
        (weekly_df['weeks_training'] > BEGINNER_WEEKS) & 
        (weekly_df['weeks_training'] <= INTERMEDIATE_WEEKS)
    ]
    advanced = weekly_df[weekly_df['weeks_training'] > INTERMEDIATE_WEEKS]

    return beginner, intermediate, advanced

def correlation(df, col1, col2):
    """
    Calculate correlation.
    
    Args:
        df: DataFrame
        col1: First column name
        col2: Second column name
        
    Returns:
        Correlation coefficient or np.nan if insufficient data
    """
    if len(df) > 1:
        return df[col1].corr(df[col2])
    return np.nan

def format_correlation(corr):
    """Format correlation coefficient for display."""
    if np.isnan(corr):
        return "N/A"
    return f"{corr:+.3f}"

def create_comparison_plot(weekly_df, vis_folder):
    """
    Create side-by-side comparison of old vs new correlation method.
    
    Args:
        weekly_df: DataFrame with weekly metrics
        vis_folder: Path to save visualization
    """
    corr_old = weekly_df['volume_total'].corr(weekly_df['prs'])
    corr_new = weekly_df['volume_training'].corr(weekly_df['prs'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Old method 
    ax1.scatter(weekly_df['volume_total'], weekly_df['prs'],
                s=80, alpha=0.6, color='#FF6868', edgecolors='black')
    coeffs = np.polyfit(weekly_df['volume_total'], weekly_df['prs'], 1)
    trend_fn = np.poly1d(coeffs)
    x_line = np.linspace(weekly_df['volume_total'].min(),
                         weekly_df['volume_total'].max(), 100)
    ax1.plot(x_line, trend_fn(x_line), "r--", linewidth=2)
    ax1.set_xlabel('Total Volume (includes PR sets)', fontsize=11, 
                   fontweight='bold')
    ax1.set_ylabel('PRs Hit', fontsize=11, fontweight='bold')
    ax1.set_title(f'OLD: Total Volume vs PRS\nr = {corr_old:+.3f}',
                  fontsize=12, fontweight='bold', color='#FF6B6B')
    ax1.grid(True, alpha=0.3)

    # Right: New method 
    ax2.scatter(weekly_df['volume_training'], weekly_df['prs'],
                s=80, alpha=0.6, color='#51CF66', edgecolors='black')
    coeffs = np.polyfit(weekly_df['volume_training'], weekly_df['prs'], 1)
    trend_fn = np.poly1d(coeffs)
    x_line = np.linspace(weekly_df['volume_training'].min(), 
                          weekly_df['volume_training'].max(), 100)
    ax2.plot(x_line, trend_fn(x_line), "g--", linewidth=2)
    ax2.set_xlabel('Training Volume (excludes PR sets)', fontsize=11, 
                   fontweight='bold')
    ax2.set_ylabel('PRs Hit', fontsize=11, fontweight='bold')
    ax2.set_title(f'NEW: Training Volume vs PRs\nr = {corr_new:+.3f}', 
                  fontsize=12, fontweight='bold', color='#51CF66')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(vis_folder, 'reverse_causation_correction.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.clf()

    print("✓ Created visualizations/reverse_causation_correction.png")
    
def create_lagged_plot(weekly_df, vis_folder):
    """
    Create scatter plot of lagged correlation (last week → this week).
    
    Args:
        weekly_df: DataFrame with lagged features
        vis_folder: Path to save visualization
    """
    plot_data = weekly_df.dropna(subset=['volume_prev_week'])
    corr_lagged = plot_data['volume_prev_week'].corr(plot_data['prs'])

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(plot_data['volume_prev_week'], plot_data['prs'],
               s=80, alpha=0.6, color='#2E86AB', edgecolors='black')
    
    z = np.polyfit(plot_data['volume_prev_week'], plot_data['prs'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_data['volume_prev_week'].min(), 
                          plot_data['volume_prev_week'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2)
    
    ax.set_xlabel('Previous Week Training Volume', fontsize=12, fontweight='bold')
    ax.set_ylabel('This Week PRs', fontsize=12, fontweight='bold')
    ax.set_title(f'Does Last Week\'s Volume Predict This Week\'s PRs?\n'
                 f'r = {corr_lagged:+.3f}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(vis_folder, 'lagged_volume_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.clf()

    print("✓ Created visualizations/lagged_volume_correlation.png")

def save_corrected_data(weekly_df, vis_folder):
    """
    Save corrected weekly metrics to CSV.
    
    Args:
        weekly_df: DataFrame with all weekly metrics
        vis_folder: Path to visualization folder (to find outputs folder)
    """
    summary_path = os.path.join(os.path.dirname(vis_folder), 'outputs')
    os.makedirs(summary_path, exist_ok=True)
    
    corrected_file = os.path.join(summary_path, 'weekly_summary_corrected.csv')
    weekly_save = weekly_df.copy()
    weekly_save.index = weekly_save.index.astype(str)
    weekly_save.to_csv(corrected_file)
    
    print(f"✓ Saved corrected data to outputs/weekly_summary_corrected.csv")

# ===== MAIN ANALYSIS =====

def main():

    print("=" * 80)
    print("ADVANCED VOLUME ANALYSIS")
    print("Fixing reverse causation: PRs cause volume increases")
    print("=" * 80)
    print()

    # Setup
    vis_folder = setup_output_folder()

    # Load and prepare data
    print("Loading your workout data...")
    df = load_workout_data(DATA_PATH)

    print("Identifying PR sets...")
    df = identify_pr_sets(df)

    # Calculate weekly metrics
    print("\n" + "=" * 80)
    print("CALCULATING VOLUME THREE WAYS")
    print("=" * 80)
    
    weekly_summary = calculate_weekly_metrics(df)
    weekly_summary = add_lagged_features(weekly_summary)

    print(f"\n✓ Analyzed {len(weekly_summary)} weeks")
    print(f"\nAverage weekly metrics:")
    print(f"  Total volume:    {weekly_summary['volume_total'].mean():,.0f}" 
          f"lbs")
    print(f"  Training volume: {weekly_summary['volume_training'].mean():,.0f}" 
          f"lbs (excludes PR sets)")
    print(f"  Difference:      {(weekly_summary['volume_total'] - 
                                 weekly_summary['volume_training']).mean()
                                 :,.0f} lbs")
    print(f"  Sets per week:   {weekly_summary['sets'].mean():.0f}")
    print(f"  PRs per week:    {weekly_summary['prs'].mean():.1f}")

    # Correlational analysis
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    corr_old = weekly_summary['volume_total'].corr(weekly_summary['prs'])
    corr_new = weekly_summary['volume_training'].corr(weekly_summary['prs'])
    corr_sets = weekly_summary['sets'].corr(weekly_summary['prs'])

    print(f"\n🔴 OLD METHOD (flawed):")
    print(f"  Total volume vs PRs: {corr_old:+.3f}")
    print(f"  Problem: PRs increase volume (reverse causation)")
    
    print(f"\n✅ NEW METHOD (corrected):")
    print(f"  Training volume vs PRs: {corr_new:+.3f}")
    print(f"  (Excludes PR sets - measures training effect only)")
    
    print(f"\n✅ ALTERNATIVE METHOD:")
    print(f"  Set count vs PRs: {corr_sets:+.3f}")
    print(f"  (Sets don't automatically increase with PRs)")

    print(f"\n💡 INTERPRETATION:")
    if abs(corr_new) < abs(corr_old):
        print(f"  My corrected correlation is LOWER ({corr_new:+.3f} vs "
              f"{corr_old:+.3f})")
        print(f"  This confirms reverse causation was inflating the "
              f"relationship!")
        print(f"  The TRUE effect of my training volume on PRs is "
              f"{corr_new:+.3f}")
    else:
        print(f"  My corrected correlation is similar or higher")
        print(f"  Training volume genuinely matters for my PRs!")

    # Lagged analysis
    print("\n" + "=" * 80)
    print("LAGGED ANALYSIS: Does Last Week's Volume Predict This Week's PRs?")
    print("=" * 80)

    corr_lagged = (weekly_summary['volume_prev_week']
                   .corr(weekly_summary['prs']))
    corr_sets_lagged = (weekly_summary['sets_prev_week']
                        .corr(weekly_summary['prs']))
    
    print(f"\nCurrent week training volume → current week PRs: "
          f"{corr_new:+.3f}")
    print(f"Previous week training volume → current week PRs: "
          f"{corr_lagged:+.3f}")
    
    print(f"\nCurrent week sets → current week PRs: {corr_sets:+.3f}")
    print(f"Previous week sets → current week PRs: {corr_sets_lagged:+.3f}")

    print(f"\n💡 INTERPRETATION:")
    if abs(corr_lagged) > 0.3:
        print(f"  Last week's volume DOES predict my PRs this week!")
        print(f"  High volume → adaptation → PRs next week")
    else:
        print(f"  Weak lagged correlation")
        print(f"  My PRs seem more immediate (same week) or random")

    # Stratified analysis
    print("\n" + "=" * 80)
    print("STRATIFIED ANALYSIS")
    print("=" * 80)
    
    beginner, intermediate, advanced = split_by_training_phase(weekly_summary)
    
    print(f"\nPhase distribution:")
    print(f"  Beginner (0-6 mo):     {len(beginner)} weeks")
    print(f"  Intermediate (6-12 mo): {len(intermediate)} weeks")
    print(f"  Advanced (12+ mo):      {len(advanced)} weeks")

    # Calculate phase correlations
    corr_beg_old = correlation(beginner, 'volume_total', 'prs')
    corr_beg_new = correlation(beginner, 'volume_training', 'prs')
    corr_int_old = correlation(intermediate, 'volume_total', 'prs')
    corr_int_new = correlation(intermediate, 'volume_training', 'prs')
    corr_adv_old = correlation(advanced, 'volume_total', 'prs')
    corr_adv_new = correlation(advanced, 'volume_training', 'prs')

    print(f"\n📊 OLD METHOD (with reverse causation):")
    print(f"  Beginner:     {format_correlation(corr_beg_old)}")
    print(f"  Intermediate: {format_correlation(corr_int_old)}")
    print(f"  Advanced:     {format_correlation(corr_adv_old)}")
    
    print(f"\n✅ NEW METHOD (corrected):")
    print(f"  Beginner:     {format_correlation(corr_beg_new)}")
    print(f"  Intermediate: {format_correlation(corr_int_new)}")
    print(f"  Advanced:     {format_correlation(corr_adv_new)}")

    print(f"\n💡 INTERPRETATION:")
    beg_drop = corr_beg_old - corr_beg_new
    adv_drop = corr_adv_old - corr_adv_new
    if adv_drop > beg_drop:
        print(f"  Reverse causation worsens as I advance — each PR")
        print(f"  is heavier, inflating volume more than as a beginner")
    if corr_beg_new > corr_adv_new:
        print(f"  Volume matters less and less as I advance:")
        print(f"  Beginner: {format_correlation(corr_beg_new)} → "
              f"Advanced: {format_correlation(corr_adv_new)}")
    if not np.isnan(corr_adv_new) and abs(corr_adv_new) < 0.2:
        print(f"  At my current level, volume barely predicts PRs")
        print(f"  My PRs are likely driven by recovery, peaking, and")
        print(f"  readiness — not just how much volume I did")

    # Visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    create_comparison_plot(weekly_summary, vis_folder)
    create_lagged_plot(weekly_summary, vis_folder)
    
    # Save data
    print("\n" + "=" * 80)
    print("SAVING CORRECTED ANALYSIS")
    print("=" * 80)
    
    save_corrected_data(weekly_summary, vis_folder)

    # Final insights
    print("\n" + "=" * 80)
    print("CORRECTED INSIGHTS")
    print("=" * 80)
    
    print(f"\n🔬 THE FIX:")
    print(f"  Problem: PRs increase weight → increase volume")
    print(f"  Solution: Exclude PR sets from volume calculation")
    print(f"  Result: TRUE training effect on PRs")
    
    print(f"\n📊 CORRECTED CORRELATIONS:")
    print(f"  Old (flawed):  {corr_old:+.3f}")
    print(f"  New (correct): {corr_new:+.3f}")
    print(f"  Difference:    {abs(corr_old - corr_new):.3f}")

    if abs(corr_new) < abs(corr_old):
        print(f"\n  ✓ Reverse causation confirmed!")
        print(f"  ✓ The TRUE relationship is weaker than it appeared")
    
    print(f"\n💡 ADVANCED ANALYSIS :")
    if not np.isnan(corr_adv_new):
        print(f"  Advanced phase correlation: {corr_adv_new:+.3f}")
        if abs(corr_adv_new) > 0.4:
            print(f"  ✓ Training volume strongly affects PRs at your level")
        elif abs(corr_adv_new) > 0.2:
            print(f"  ✓ Training volume moderately affects PRs")
        else:
            print(f"  ⚠️  Training volume weakly related to PRs")
            print(f"  Focus on: intensity, technique, recovery")
            
    print(f"\n🎯 LAGGED EFFECT:")
    print(f"  Last week's volume → This week's PRs: {corr_lagged:+.3f}")
    if abs(corr_lagged) > abs(corr_new):
        print(f"  ✓ Delayed effect! Volume needs time to cause adaptation")
    elif abs(corr_lagged) > 0.3:
        print(f"  ✓ Some delayed effect present")
    else:
        print(f"  ⚠️  Weak delayed effect")
    
    print("\n" + "=" * 80)
    print("ADVANCED VOLUME ANALYSIS COMPLETE! 🎯")
    print("=" * 80)

# ===== RUN ANALYSIS =====

if __name__ == '__main__':
    main()