import pandas as pd
from src.preprocessing import prepare_data
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.getcwd())

df = pd.read_csv('data/tox21.csv.gz', compression='gzip')
data = prepare_data(df, split_type='scaffold')

total = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
print(f"\n--- Resulting Scaffold Split Ratio ---")
print(f"Train: {len(data['X_train']):>4} samples ({len(data['X_train'])/total:>6.2%})")
print(f"Val:   {len(data['X_val']):>4} samples ({len(data['X_val'])/total:>6.2%})")
print(f"Test:  {len(data['X_test']):>4} samples ({len(data['X_test'])/total:>6.2%})")
