import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#Output Folder
script_dir = os.path.dirname(os.path.abspath(__file__))
vis_folder = os.path.join(script_dir, 'visualizations')
os.makedirs(vis_folder, exist_ok=True)
print(f"Saving visualizations to: {vis_folder}")

#Data
df = pd.read_csv('workout_data.csv')
df['date'] = pd.to_datetime(df['start_time'], format = '%d %b %Y, %H:%M')
weighted_sets = df[df['weight_lbs'].notna() & (df['weight_lbs'] > 0)].copy()

print(f"Loaded {len(df)} total sets")
print(f"Analyzing {len(weighted_sets)} weighted sets\n")

def save_path(file_name):
    return os.path.join(vis_folder, file_name)

def plot_exercise_progression(exercise_name, color='#2E86AB', \
                              save_name=None):
    """
    PLOT PR progression for a single exercise
    
    :param exercise_name: Name of the exercise to plot
    :param color: Color for the line (hex code or name)
    :param save_name: Optional filename to save (without .png)
    """

    #Filter and prepare data
    data = df[df['exercise_title'] == exercise_name]

    if len(data) == 0:
        print(f"No data found for {exercise_name}")
        return None
    
    #Sorting by date and calculating running max
    data = data.sort_values('date')
    data['running_max'] = data['weight_lbs'].cummax()
    
    #Finding PR changes
    pr_changes = data[data['running_max'] != data['running_max'].shift()]

    #Plotting the progression
    plt.figure(figsize=(12, 6))

    #Line Chart
    plt.plot(pr_changes['date'], pr_changes['running_max'],
             marker='o', markersize=8, linewidth=2.5,
             color=color, label=f"{exercise_name} PR")
    
    #Labels
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Weight (lbs)', fontsize=12, fontweight='bold')
    plt.title(f"{exercise_name} - PR Progression Over Time")

    #Layout
    plt.grid(True, alpha=0.3, linestyle= '--')
    plt.legend(fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    #Saving File
    filename = save_name if save_name else \
               exercise_name.lower().replace(' ', '_')
    plt.savefig(save_path(f"{filename}_progression.png"), \
                dpi = 150, bbox_inches='tight')
    print(f"Created {filename}_progression.png")

    plt.clf()
    return pr_changes

#Plotting Bench, Squat, Lat Pulldown
exercises_to_plot = [
    ('Bench Press (Barbell)', '#2E86AB'),
    ('Leg Extension', '#A23B72'),
    ('Lat Pulldown (Cable)', '#F18F01')
]

for exercise, color in exercises_to_plot:
    plot_exercise_progression(exercise, color)

#Top 10 PRS Bar Chart
prs = weighted_sets.groupby('exercise_title')['weight_lbs'].max()\
      .sort_values(ascending=False)
top_10 = prs.head(10)

plt.figure(figsize=(16,6))
bars = plt.barh(range(len(top_10)), top_10.values, color='#2E86AB')

#Bar Podium Colors
bars[0].set_color("#FFD700")
if len(bars) > 1:
    bars[1].set_color('#C0C0C0')
if len(bars) > 2:
    bars[2].set_color('#CD7F32')

#Labels
plt.yticks(range(len(top_10)), top_10.index, fontsize=10)
plt.gca().invert_yaxis()
plt.xlabel('PR Weight (lbs)', fontsize=12, fontweight='bold')
plt.title('Top 10 Heaviest Personal Records', fontsize=14, fontweight='bold', \
          pad=20)
plt.tight_layout()

for i,v in enumerate(top_10.values):
    plt.text(v + 5, i, f'{v:.0f} lbs', va='center', fontsize=9)

#Saving Bar Chart
plt.savefig(save_path("top_10_prs.png"), dpi = 150, bbox_inches='tight')
print("Created top_10_prs.png")
print(f"  My top 3:")
print(f"  🥇 {top_10.index[0]}: {top_10.values[0]:.0f} lbs")
if len(top_10) > 1:
    print(f"  🥈 {top_10.index[1]}: {top_10.values[1]:.0f} lbs")
if len(top_10) > 2:
    print(f"  🥉 {top_10.index[2]}: {top_10.values[2]:.0f} lbs")
plt.clf()

compare_exercises = [
    ('Bench Press (Barbell)', '#2E86AB', 'o'),
    ('Bench Press (Dumbbell)', '#F18F01', 'o'),
    ('Bench Press (Smith Machine)', '#A23B72', 'o'),
    ('Incline Bench Press (Smith Machine)', '#E94F37', 'o'),
    ('Incline Bench Press (Dumbbell)', '#44BBA4', 'o')
]

plt.figure(figsize=(14,7))

for exercise, color, marker in compare_exercises:
    data = df[df['exercise_title'] == exercise].copy()
    if len(data) > 0:
        data = data.sort_values('date')
        data['running_max'] = data['weight_lbs'].cummax()
        pr_changes = data[data['running_max'] != data['running_max'].shift()]

        plt.plot(pr_changes['date'], pr_changes['running_max'],
                 marker=marker, markersize=7, linewidth=2,
                 color=color, label=exercise, alpha=0.8
                 )
        
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Weight (lbs)', fontsize=12, fontweight='bold')
plt.title('Pressing Exercises - PR Comparison', fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig(save_path('pressing_comparison.png'), \
            dpi=150, bbox_inches='tight')
print("Created pressing_comparison.png")
print("  Compares different pressing movements")
plt.clf()