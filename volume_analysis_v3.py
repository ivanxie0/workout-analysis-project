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
DATA_PATH = '/mnt/project/workout_data.csv'
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
    weekly-df['sets_prev_week'] = weekly_df['sets'].shift(1)
    return weekly_df

def split_by_training_phase(weekly_df):
    beginner = weekly_df[weekly_df['weeks_training'] <= BEGINNER_WEEKS]
    intermediate = weekly_df[
        (weekly_df['weeks_training'] > BEGINNER_WEEKS) & 
        (weekly_df['weeks_training'] <= INTERMEDIATE_WEEKS)
    ]
    advanced = weekly_df[weekly_df['weeks_training'] > INTERMEDIATE_WEEKS]

    return beginner, intermediate, advanced

