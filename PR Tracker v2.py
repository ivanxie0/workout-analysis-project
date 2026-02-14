import pandas as pd

print("=" * 60)
print("TIME TRACKING")
print("=" * 60)

#LOADING AND CONVERTING DATES
print("\n" + "=" * 80)
print("CONVERTING DATES TO DATETIME OBJECTS")
print("=" * 80)

df = pd.read_csv('workout_data.csv')
weighted_sets = df[df['weight_lbs'].notna() & (df['weight_lbs'] > 0)].copy()

print("\nOriginal date format (string):")
print(f"  {weighted_sets['start_time'].iloc[0]}")
print(f"  Type: {type(weighted_sets['start_time'].iloc[0])}")

weighted_sets['date'] = pd.to_datetime(weighted_sets['start_time'], \
                                       format = '%d %b %Y, %H:%M')

print("\nNew datetime format:")
print(f"  {weighted_sets['date'].iloc[0]}")
print(f"  Type: {type(weighted_sets['date'].iloc[0])}")

#FINDING DATE WHEN PR WAS HIT

#EXAMPLE WITH BENCH PRESS
bench_data = weighted_sets[weighted_sets['exercise_title'] == \
                           'Bench Press (Barbell)'].copy()
print(f"\nExample: Bench Press (Barbell)")
print(f"  Total sets: {len(bench_data)}")
print(f"  Max weight: {bench_data['weight_lbs'].max()} lbs")
print(f"\n Sets at max weight:")
pr_sets = bench_data[bench_data['weight_lbs'] == bench_data['weight_lbs'].max()]
print(pr_sets[['date', 'weight_lbs', 'reps']].head())

""" 
This function receives ONE group (all sets for one exercise).
It returns information about that exercise's PR.
"""

def get_pr_details(group):
    max_weight = group['weight_lbs'].max()
    pr_rows = group[group['weight_lbs'] == max_weight]

    first_pr = pr_rows['date'].min()
    last_pr = pr_rows['date'].max()
    today = pd.Timestamp.now()

    return pd.Series({
        'PR_weight_lbs': max_weight,
        'first_achieved' : first_pr.strftime('%d %b %Y'),
        'last_achieved' : last_pr.strftime('%d %b %Y'),
        'times_hit' : len(pr_rows),
        'days_since' : (today - last_pr).days
    })

print("\n" * 2)
print("=" * 80)
print("APPLYING PR DETAILS FUNCTION ON ALL EXERCISES")
print("=" * 80)

prs = weighted_sets.groupby('exercise_title')\
    .apply(get_pr_details).reset_index()

print("\nResult")
print(prs.head())

print(f"\n We now have PR details for all {len(prs)} exercises!")

#FILTERING

#Recent PRS (within 30 days)
recent = prs[prs['days_since'] <= 30]
print(f"\nPRS hit in last 30 days: {len(recent)}")
print(f"  Condition : days_since <= 30")

#Stale PRS (over 180 days)
stale = prs[prs['days_since'] > 180]
print(f"\n PRS not hit in 6+ months: {len(stale)}")
print(f"  Condition : days_since > 180")

#Consistent PRS (hit 10+ times)
consistent = prs[prs['times_hit'] >= 10]
print(f"\nPRs hit 10+ times: {len(consistent)}")
print(f"  Condition: times_hit >= 10")


