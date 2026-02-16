import pandas as pd

df = pd.read_csv('workout_data.csv')
# NUMBER OF SETS
print(f"Loaded {len(df)} sets from your workouts!\n")

# TITLE WITH DIVIDERS
print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)

# START DATE TO END DATE
print(f"Data range: {df['start_time'].min()} to {df['start_time'].max()}")

# NUMBER OF UNIQUE EXERCISES
print(f"Unique exercises: {df['exercise_title'].nunique()}")

# SAMPLE DATA
print("Sample of data")
print(df[['exercise_title', 'weight_lbs', 'reps', 'start_time']].head(10))

# FILTERING DATA
print("\n" + "=" * 60)
print("FINDING PERSONAL RECORDS")
print("=" * 60)

weighted_sets = df[df['weight_lbs'].notna() & df['weight_lbs'] > 0]
print(f"Analyzing {len(weighted_sets)} weighted sets...\n")

# GROUPING AND FINDING MAX WEIGHT
prs = weighted_sets.groupby('exercise_title').agg({
    'weight_lbs': 'max',
    'start_time': 'max'
})

prs = prs.reset_index()

#RENAME COLUMNS
prs.columns = ['Exercise', 'PR Weight (lbs)', 'Last Performed']

#SORTING VALUES TO DESCENDING
prs = prs.sort_values('PR Weight (lbs)', ascending=False)

print(f"Found PRS for {len(prs)} exercises!\n")

#DISPLAY RESULTS
print("TOP 10 HEAVIEST LIFTS:")
print("-" * 60)
print(prs.head(10).to_string(index=False))

print("\n\nALL PERSONAL RECORDS:")
print("=" * 60)
print(prs.to_string(index=False))
print("=" * 60)

print(f"My heaviest lift ever: {prs.iloc[0]['Exercise']} -\
     {prs.iloc[0]['PR Weight (lbs)']} lbs")

print(f"Average PR weight: {prs['PR Weight (lbs)'].mean():.1f} lbs")

print(f"Total exercises tracked: {len(prs)}")
prs.to_csv('my_prs.csv', index=False)