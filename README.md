# How Far Does the Model Hold?

**Robustness of Multihorizon Battery Failure Intelligence Under Realistic Operating Conditions**

This project evaluates the robustness of a multihorizon battery failure prediction model when operating conditions deviate from clean laboratory data. It tests whether an XGBoost hazard model remains reliable under partial cycling, temperature noise, and irregular rest periods.

Key findings:
- Calibration degrades ~9x (ECE from 0.03 to 0.28) under even mild perturbation
- Recalibration on 10% of operational data recovers 75–81% of the gap
- Three operating zones: Safe (clean lab), Warning (recalibrated), Unsafe (direct deployment)

## Structure

| Path | Purpose |
|------|---------|
| `src/` | Robustness study source: synthetic data gen, training, evaluation, plotting, tables |
| `paper/` | LaTeX paper source and PDF |
| `data/` | Clean NASA reference dataset |
| `results/` | Robustness evaluation metrics |
| `figs/` | Generated figures |
| `tables/` | Paper tables (CSV, PNG) |
| `battery_sens/` | Original multihorizon hazard learning framework (journal paper) |

## Usage

```bash
# 1. Generate perturbed datasets from raw NASA .mat files
python src/synthetic_data.py

# 2. Train model on clean data
python src/train_full_model.py

# 3. Evaluate robustness across all perturbations
python src/run_robustness.py

# 4. Generate figures and tables
python src/plot_robustness.py
python src/build_robustness_tables.py
```

## Dependencies

Python 3, numpy, pandas, scikit-learn, xgboost, matplotlib, joblib, scipy.

## Citation

T. A. Shikdar and H. Laaksonen, "How Far Does the Model Hold? Robustness of Multihorizon Battery Failure Intelligence Under Realistic Operating Conditions," 2026.
