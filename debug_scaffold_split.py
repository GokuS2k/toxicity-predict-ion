from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
import logging

# Configure minimal logging to avoid noise
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return "INVALID"

# 1. Load data
df = pd.read_csv('data/tox21.csv.gz', compression='gzip').dropna(subset=['smiles'])
smiles = df['smiles'].values
n = len(smiles)

# 2. Map every molecule to a scaffold
scaffold_to_indices = {}
for i, smi in enumerate(smiles):
    scf = get_scaffold(smi)
    scaffold_to_indices.setdefault(scf, []).append(i)

# 3. The "Why": Size-based Sorting
# This is explicitly what the code does: sorts groups by size largest -> smallest
scaffold_groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)

# 4. Reproducible Shuffle
# This determines which specific scaffolds go where
random_state = 42
rng = np.random.RandomState(random_state)
order = rng.permutation(len(scaffold_groups))
shuffled_groups = [scaffold_groups[i] for i in order]

# 5. Greedy Assignment
train_target = 0.8 * n
val_target = 0.1 * n

train_indices = []
val_indices = []
test_indices = []

scaffold_assignment = {} # To track which scaffold went where

for i, group in enumerate(shuffled_groups):
    # Find the scaffold string for this group unit (using first member)
    scf_str = get_scaffold(smiles[group[0]])
    
    if len(train_indices) < train_target:
        train_indices.extend(group)
        scaffold_assignment[scf_str] = "TRAIN"
    elif len(val_indices) < val_target:
        val_indices.extend(group)
        scaffold_assignment[scf_str] = "VAL"
    else:
        test_indices.extend(group)
        scaffold_assignment[scf_str] = "TEST"

# 6. Report findings
print(f"Total Scaffolds: {len(scaffold_groups)}")
stats = pd.Series(scaffold_assignment).value_counts()
print("\nScaffold Counts per Split:")
print(stats)

print("\n--- Examples of Scaffolds assigned to TEST ---")
test_scafs = [s for s, split in scaffold_assignment.items() if split == "TEST"]
for s in test_scafs[:10]:
    count = len(scaffold_to_indices[s])
    print(f"  Scaffold: {s if s else 'Non-Ring'} (Used in {count} molecules)")

print("\n--- Examples of Scaffolds assigned to TRAIN ---")
train_scafs = [s for s, split in scaffold_assignment.items() if split == "TRAIN"]
for s in train_scafs[:5]:
    count = len(scaffold_to_indices[s])
    print(f"  Scaffold: {s if s else 'Non-Ring'} (Used in {count} molecules)")
