import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Output Folder
script_dir = os.path.dirname(os.path.abspath(__file__))
vis_folder = os.path.join(script_dir, 'visualizations')
os.makedirs(vis_folder, exist_ok=True)

# Loading and Preparing Data
df = pd.read_csv('workout_data.csv')
df['date'] = pd.to_datetime(df['start_time'], format='%d %b %Y, %H:%M')

# Calculating volume for each set
df['volume'] = df['reps'] * df['weight_lbs'] 

print(f"✓ Loaded {len(df)} total sets")
print(f"  Total volume lifted all-time: {df['volume'].sum():,.0f} lbs")

# Calculating weekly metrics
df['week'] = df['date'].dt.to_period('W')

# Weekly volume
weekly_volume = df.groupby('week')['volume'].sum()

# Weekly sets
weekly_sets = df.groupby('week').size()

# Weekly PRS
df_sorted = df.sort_values(['exercise_title', 'date'])
df_sorted['running_max'] = df_sorted.groupby('exercise_title')['weight_lbs']\
         .cummax()
df_sorted['is_pr'] = df_sorted['running_max'] != \
                     df_sorted.groupby('exercise_title')['running_max'].shift()
df_sorted['week'] = df_sorted['date'].dt.to_period('W')
weekly_prs = df_sorted[df_sorted['is_pr']].groupby('week').size()

# Combine into weekly summary
weekly_summary = pd.DataFrame({
    'volume': weekly_volume,
    'sets': weekly_sets,
    'prs': weekly_prs
}).fillna(0)

# Calculating rolling averages
weekly_summary['volume_4wk_avg'] = weekly_summary['volume'].rolling(window=4)\
              .mean()
weekly_summary['volume_8wk_avg'] = weekly_summary['volume'].rolling(window=8)\
              .mean()

print(f"✓ Analyzed {len(weekly_summary)} weeks of training")
print(f"  Average weekly volume: {weekly_summary['volume'].mean():.0f} lbs")
print(f"  Average weekly sets: {weekly_summary['sets'].mean():.0f}")
print(f"  Average PRS per week: {weekly_summary['prs'].mean():.1f}")

# VISUALIZATION 1: WEEKLY VOLUME OVER TIME
print("=" * 80)
print("VISUALIZATION 1: WEEKLY VOLUME TRENDS")
print("=" * 80)
plt.figure(figsize=(14, 7))

# Plotting actual weekly volume
plt.plot(weekly_summary.index.astype(str), weekly_summary['volume'],
         marker='o', markersize=4, linewidth=1, alpha=0.5,
         color='#2E86AB', label='Weekly Volume')

# Plotting 4-week rolling average
plt.plot(weekly_summary.index.astype(str), weekly_summary['volume_4wk_avg'],
         linewidth=3, color='#F18F01', label='4-Week Average')

# Plotting 8-week rolling average
plt.plot(weekly_summary.index.astype(str), weekly_summary['volume_8wk_avg'],
         linewidth=3, color='#A23B72', label='8-Week Average', 
         linestyle ='--')

plt.xlabel('Week', fontsize=12, fontweight='bold')
plt.ylabel('Total Volume (lbs)', fontsize=12, fontweight='bold')
plt.title('Weekly Training Volume Over Time', fontsize=14, fontweight='bold',
          pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45, ha='right')

ax = plt.gca()
labels = ax.get_xticklabels()
for i, label in enumerate(labels):
    if i % 4 != 0:
        label.set_visible(False)

plt.tight_layout()
save_path = os.path.join(vis_folder, 'weekly_volume_trends.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print("✓ Created visualizations/weekly_volume_trends.png")
plt.clf()
print()

# VISUALIZATION 2: VOLUME VS PRS SCATTERPLOT
print("=" * 80)
print("VISUALIZATION 2: VOLUME VS PRs CORRELATION")
print("=" * 80)

# Calculating correlation
correlation = weekly_summary['volume'].corr(weekly_summary['prs'])

plt.figure(figsize=(10,7))
plt.scatter(weekly_summary['volume'], weekly_summary['prs'],
            s=100, alpha=0.6, color='#2E86AB', edgecolors='black')

# Adding trend line
z = np.polyfit(weekly_summary['volume'].dropna(),
               weekly_summary['prs'][weekly_summary['volume'].notna()], 1)
p = np.poly1d(z)
x_line = np.linspace(weekly_summary['volume'].min(), \
                     weekly_summary['volume'].max(), 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, \
         label=f'Trend Line (r={correlation:+.3f})')

plt.xlabel('Weekly Volume (lbs)', fontsize=12, fontweight='bold')
plt.ylabel('PRs Hit That Week', fontsize=12, fontweight='bold')
plt.title('Relationship Between Volume and PR Progress', fontsize=14, 
          fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = os.path.join(vis_folder, 'volume_vs_prs_correlation.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print("✓ Created visualizations/volume_vs_prs_correlation.png")
print(f"  Correlation coefficient: {correlation:+.3f}")

if abs(correlation) < 0.3:
    print("  → Weak correlation")
elif abs(correlation) < 0.7:
    print("  → Moderate correlation")
else:
    print("  → Strong correlation")

if correlation > 0:
    print("  → Positive: More volume tends to mean more PRs!")
else:
    print("  → Negative: More volume tends to mean fewer PRs \
          (possible overtraining?)")
plt.clf()
print()

# VISUALIZATION 3: VOLUME BY EXERCISE
print("=" * 80)
print("VISUALIZATION 3: TOP EXERCISES BY TOTAL VOLUME")
print("=" * 80)

# Calculate total volume per exercise
exercise_volume = df.groupby('exercise_title')['volume'].sum()\
                  .sort_values(ascending=False)
top_15 = exercise_volume.head(15)

plt.figure(figsize=(14,8))
bars = plt.barh(range(len(top_15)), top_15.values, color='#2E86AB', \
                edgecolor='black')

# Color top 3
bars[0].set_color('#FFD700')
if len(bars) > 1:
    bars[1].set_color('#C0C0C0')  # Silver
if len(bars) > 2:
    bars[2].set_color('#CD7F32')  # Bronze

plt.yticks(range(len(top_15)), top_15.index, fontsize=10)
plt.gca().invert_yaxis()
plt.xlabel('Total Volume (lbs)', fontsize=12, fontweight='bold')
plt.title('Top 15 Exercises by Total Volume Lifted', fontsize=14, \
          fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_15.values):
    plt.text(v + max(top_15.values)*0.01, i, f'{v:,.0f} lbs',
             va='center', fontsize=9)
    
plt.xlim(0, max(top_15.values) * 1.15)
plt.tight_layout()
save_path = os.path.join(vis_folder, 'top_exercises_by_volume.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print("✓ Created visualizations/top_exercises_by_volume.png")
print(f"  Top 3 volume exercises:")
print(f"  🥇 {top_15.index[0]}: {top_15.values[0]:,.0f} lbs")
if len(top_15) > 1:
    print(f"  🥈 {top_15.index[1]}: {top_15.values[1]:,.0f} lbs")
if len(top_15) > 2:
    print(f"  🥉 {top_15.index[2]}: {top_15.values[2]:,.0f} lbs")

plt.clf()
print()