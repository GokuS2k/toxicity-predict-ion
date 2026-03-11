import pandas as pd
import json

df = pd.read_csv('data/tox21.csv.gz', compression='gzip')
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print("\nColumn Names:")
print(df.columns.tolist())

print("\nExample Row (First Row):")
first_row = df.iloc[0].to_dict()
for col, val in first_row.items():
    print(f"  {col}: {val}")
