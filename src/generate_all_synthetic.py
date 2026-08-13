"""Generate all synthetic datasets for all (severity, seed) combinations needed
by the fixed pipeline."""
import os, sys
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from synthetic_data import generate_synthetic_dataset

SEVERITIES = [1, 2, 3, 4]
# Eval seeds + DR-train seed (2024, distinct from eval seeds)
SEEDS = [42, 123, 456, 789, 101112, 2024]

for s in SEVERITIES:
    for seed in SEEDS:
        print(f"\n=== severity={s}, seed={seed} ===")
        generate_synthetic_dataset(s, seed=seed)

print("\nAll synthetic datasets generated.")
