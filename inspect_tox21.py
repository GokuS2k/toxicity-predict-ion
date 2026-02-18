import pandas as pd

# Load the compressed CSV file directly
try:
    df = pd.read_csv('data/tox21.csv.gz', compression='gzip')
    print("Successfully loaded 'data/tox21.csv.gz'")
    print("-" * 30)
    print(f"Shape: {df.shape}")
    print("-" * 30)
    print("Columns:")
    print(df.columns.tolist())
    print("-" * 30)
    print("First 5 rows:")
    print(df.head())
except Exception as e:
    print(f"Error loading file: {e}")
