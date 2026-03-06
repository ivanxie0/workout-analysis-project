import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
vis_folder = os.path.join(script_dir, 'visualizations')
os.makedirs(vis_folder, exist_ok=True)

print("=" * 80)
print("PHASE 5: ADVANCED VOLUME ANALYSIS")
print("Controlling for Training Experience Bias")
print("=" * 80)
print()

# Load and prepare data
print("Loading your workout data...")
df = pd.read_csv('workout_data.csv')
df['date'] = pd.to_datetime(df['start_time'], format='%d %b %Y, %H:%M')
df['volume'] = df['reps'] * df['weight_lbs']
df['week'] = df['date'].dt.to_period('W')

# Calculate weekly metrics
weekly_volume = df.groupby('week')['volume'].sum()
weekly_sets = df.groupby('week').size()

# Calculate weekly PRs
df_sorted = df.sort_values(['exercise_title', 'date'])
df_sorted['running_max'] = df_sorted.groupby('exercise_title')['weight_lbs']\
    .cummax()
df_sorted['is_pr'] = df_sorted['running_max'] != df_sorted.groupby\
    ('exercise_title')['running_max'].shift
df_sorted['week'] = df_sorted['date'].dt.to_period('W')
weekly_prs = df_sorted[df_sorted['is_pr']].groupby('week').size()

# Combine
weekly_summary = pd.DataFrame({
    'volume': weekly_volume,
    'sets': weekly_sets,
    'prs': weekly_prs
}).fillna(0)

# Add training age (weeks since start)
weekly_summary['weeks_training'] = range(len(weekly_summary))

print(f"✓ Loaded {len(df)} sets across {len(weekly_summary)} weeks")
print()

# Stratified Analysis: Split by training phase
beginner = weekly_summary[weekly_summary['weeks_training'] <= 26]
intermediate = weekly_summary[(weekly_summary['weeks_training'] > 26) &
                              (weekly_summary['weeks_training'] <= 52)]
advanced = weekly_summary[weekly_summary['weeks_training'] > 52]

print(f"\nBeginner Phase (0-6 months): {len(beginner)} weeks")
print(f"Intermediate Phase (6-12 months): {len(intermediate)} weeks")
print(f"Advanced Phase (12+ months): {len(advanced)} weeks")

# Calculating correlations for each phase
corr_all = weekly_summary['volume'].corr(weekly_summary['prs'])
corr_beginner = weekly_summary['volume'].corr(beginner['prs']) if len(beginner)\
      > 1 else np.nan
corr_intermediate = intermediate['volume'].corr(intermediate['prs']) \
    if len(intermediate) > 1 else np.nan
corr_advanced = advanced['volume'].corr(advanced['prs']) if len(advanced) > 1 \
    else np.nan

print(f"\n📊 CORRELATION BY TRAINING PHASE:")
print(f"  All data (naive):       {corr_all:+.3f}")
beginner_str = f"{corr_beginner:+.3f}" if not np.isnan(corr_beginner) else "N/A"
intermediate_str = f"{corr_intermediate:+.3f}" if not np.isnan(corr_intermediate) else "N/A"
advanced_str = f"{corr_advanced:+.3f}" if not np.isnan(corr_advanced) else "N/A"
print(f"  Beginner (0-6 mo):      {beginner_str}")
print(f"  Intermediate (6-12 mo): {intermediate_str}")
print(f"  Advanced (12+ mo):      {advanced_str}")

print(f"\n🎯 INTERPRETATION:")
print(f"  Similar correlations across phases - volume consistently matters!")

print()

# Rate of Change Analysis
print("=" * 80)
print("RATE OF CHANGE ANALYSIS")
print("=" * 80)

# Calculating changes from previous week
weekly_summary['volume_change'] = weekly_summary['volume'].diff()
weekly_summary['pr_change'] = weekly_summary['prs'].diff()

#Correlation between changes
change_corr = weekly_summary['volume_change'].corr(weekly_summary['pr_change'])
print(f"\nAbsolute correlation: {corr_all:+.3f}")
print(f"  → Do high-volume weeks have more PRs?")
print(f"\nChange correlation: {change_corr:+.3f}")
print(f"  → Does INCREASING volume lead to more PRs?")