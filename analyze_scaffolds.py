from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/tox21.csv.gz', compression='gzip')
smiles = df['smiles'].tolist()

scaffolds = []
for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        scf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scf)
    else:
        scaffolds.append("INVALID")

scaffold_series = pd.Series(scaffolds)
counts = scaffold_series.value_counts()

print(f"Total Molecules: {len(smiles)}")
print(f"Unique Scaffolds: {len(counts)}")
print("\nTop 10 Most Frequent Scaffolds:")
print(counts.head(10))

# Example of an empty scaffold (linear molecules)
empty_count = (scaffold_series == '').sum()
print(f"\nMolecules with no rings/scaffolds: {empty_count}")
