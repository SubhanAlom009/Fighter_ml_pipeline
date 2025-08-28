import pandas as pd

# Analyze Football dataset
print("=== FOOTBALL DATASET ANALYSIS ===")
df_football = pd.read_csv('Football Data Test Task.csv')
print(f"Shape: {df_football.shape}")
print(f"Columns: {list(df_football.columns)}")
print("\nSample data:")
print(df_football.head(2))

# Analyze engineered dataset to see target structure
print("\n=== ENGINEERED DATASET ANALYSIS ===")
df_engineered = pd.read_csv('Football_Feature_Engineered.csv')
print(f"Shape: {df_engineered.shape}")
print(f"Columns: {list(df_engineered.columns)}")

# Check for odds columns in original
odds_cols = [col for col in df_football.columns if 'odds' in col.lower() or 'b365' in col.lower() or 'ps' in col.lower()]
print(f"\nOdds columns found: {odds_cols}")

# Check data types
print(f"\nData types:")
print(df_football.dtypes)
