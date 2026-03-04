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