# Robustness of Multihorizon Battery Failure Intelligence

**Authors:** T. A. Shikdar, H. Laaksonen (University of Vaasa, Finland)
**Target venue:** IEEE WIECON 2026

This repository contains the code, data, and revised paper for the robustness
evaluation of a multihorizon XGBoost battery-failure hazard model under
realistic operating perturbations. The final manuscript,
[`paper/New_Paper_WIECON_2026.pdf`](./paper/New_Paper_WIECON_2026.pdf), is
built from [`scripts/build_new_paper.py`](./scripts/build_new_paper.py) and
summarizes the full evaluation: clean baseline, calibration degradation under
operational shift, the recalibration field-sample requirement (20% of a
deployment window), and the failure of domain-randomized retraining to compete.

## What's new in the final revision (2026-08)

On top of the methodological fixes below, this revision adds:

1. **Rerun with revised statistics.** The robustness evaluation and figure
   pipeline were rerun; Tables IV and VI and Sections V-B/C/D carry the
   updated numbers and confidence intervals.
2. **Prose quality pass.** The manuscript text was run through an AI-detection
   audit (`/ai-check`) and a humanization pass: zero em dashes, no pattern
   rhetoric, sentence-rhythm gates, occasional first person. All 16
   prose-number consistency checks pass.
3. **Table layout fix in the Word build.** The IEEE template body is
   two-column; tables are now built at fixed 4900-twip width with
   keep-together rows (`cantSplit` + `keepNext`) and repeating header rows, so
   no table splits or collapses across pages when rendered by LibreOffice.
4. **Paper 2 change documentation.** [`Paper2_to_final_changes.md`](./Paper2_to_final_changes.md)
   (+ the PDF rendering [`Paper2_changes.pdf`](./Paper2_changes.pdf)) is a
   replace-from/replace-to diff that ports the final content into
   `12th IEEE International Women in Engineering (Paper 2).docx`.

## What's new in this revision (FIXED VERSION)

This version of the repo fixes multiple methodological issues identified in the
original submission. A full changelog is in [`CHANGELOG.md`](./CHANGELOG.md).
The headline fixes are:

1. **No more data leakage.** Train/test split now groups by *physical battery
   ID* (B0025, B0027, etc.) instead of by the full cell string. The original
   split leaked information because the same physical battery appeared in both
   splits (re-cycled under two NASA sub-campaigns).
2. **No more in-sample calibrator fitting.** The isotonic calibrator is now
   fit on out-of-fold (OOF) predictions from 5-fold `GroupKFold` within the
   train partition, instead of on the raw training predictions.
3. **No more in-sample evaluation.** All reported metrics (ECE, AUC, Brier)
   are computed on a strictly held-out test partition (7 batteries, 163
   cycles). The original code evaluated on the full 37-cell dataset, 80% of
   which the model was trained on.
4. **Same-sample paired comparison.** `ece_cal` and `ece_recal` are both
   computed on the same held-out 90% subset (10% held out for recalibrator
   fitting). The original code computed `ece_cal` on 100% and `ece_recal` on
   90%, conflating "different calibrator" with "different evaluation sample".
5. **Proper Wilcoxon test setup.** The test is now properly paired, documented
   (n=20 pairs per horizon, one-sided, Bonferroni-corrected across 4
   horizons, α = 0.0125), and the actual p-values are reported (~10⁻⁶).
6. **Causal Vmin ablation.** A new experiment (`run_vmin_ablation.py`)
   restores *only* the minimum-voltage feature to its clean value while
   leaving all other features perturbed. This provides causal (not merely
   correlational SHAP) evidence that the Vmin distribution shift is the
   primary driver of calibration collapse. Recovery: 44–90% across horizons.
7. **No DR contamination.** The domain-randomized model is now trained on
   perturbations at seed 2024 (distinct from all 5 evaluation seeds). The
   original code trained DR on seed 42 *and* evaluated on seed 42 — in-sample
   contamination that artificially lowered DR ECE by ~5× for that one seed.
8. **Platt `C=1.0`** (sklearn default) instead of `C=9999` (unregularized,
   unstable on small samples).
9. **Fixed AUC caption.** The original Fig. 2 caption "AUC remains stable"
   was incorrect — AUC visibly declines (and is non-monotonic in the honest
   evaluation). The new caption reflects the actual behavior.
10. **Numerically stable temperature scaling.** Uses `logaddexp` instead of
    naive `exp`, with multi-start initialization.

## Honest re-evaluated numbers

Under the strictly held-out protocol, the clean-baseline numbers are
substantially weaker than originally claimed:

| Metric | Original (in-sample) | Revised (held-out) |
|--------|----------------------|--------------------|
| Clean ECE (H=20) | 0.031 | **0.231** |
| Clean AUC (H=20) | 0.985 | **0.594** |
| Perturbed ECE (S1, H=20) | 0.286 | **0.317** |
| Perturbed AUC (S1, H=20) | 0.741 | **0.616** |

The qualitative "saturation at S1" finding still holds, but the magnitude is
much smaller (a 0.09 absolute ECE increase, not a 9× jump). The recalibration
recovery and Vmin ablation findings are robust to the evaluation protocol.

## Repository structure

```
.
├── README.md                    # this file
├── CHANGELOG.md                 # detailed list of all fixes
├── Paper2_to_final_changes.md   # replace-from/replace-to diff for the WIE Paper 2 docx
├── Paper2_changes.pdf           # human-readable rendering of the diff
├── requirements.txt             # Python dependencies
├── data/
│   ├── nasa_clean_filtered.csv  # 1028 rows, 37 cell strings (33 phys. batteries)
│   ├── nasa/                    # raw .mat files (download via scripts/download_nasa.sh)
│   └── synthetic/               # perturbed CSVs (regenerated by the pipeline)
├── scripts/                     # manuscript build + data download
│   ├── build_new_paper.py       # builds the final .docx (prose, tables, OMML equations)
│   ├── download_nasa.sh         # NASA Ames dataset download
│   ├── revise_paper.py          # earlier revision script
│   └── plot_methodology.py, plot_recal_sweep.py, plot_deployment_framework.py
├── src/                         # all Python source code
│   ├── composite_label.py       # composite failure label
│   ├── synthetic_data.py        # perturbation generator
│   ├── compute_ece.py           # ECE (10 equal-width bins, last bin closed)
│   ├── train_full_model.py      # train XGBoost + OOF isotonic calibrator
│   ├── run_robustness.py        # robustness evaluation (val-only, same-sample)
│   ├── run_sample_sweep.py      # 5/10/20/50% sample sweep
│   ├── train_domain_rand.py     # train DR model (seed 2024, no contamination)
│   ├── run_domain_rand.py       # evaluate DR model
│   ├── run_vmin_ablation.py     # causal Vmin-only ablation
│   ├── bootstrap_ci.py          # paired bootstrap CIs (Table VI)
│   ├── wilcoxon_test.py         # paired Wilcoxon (Bonferroni-corrected)
│   ├── plot_*.py                # figure generation
│   ├── build_robustness_tables.py
│   └── generate_all_synthetic.py
├── results/                     # CSV outputs (regenerated by the pipeline)
├── figs/                        # PNG figures (regenerated by the pipeline)
├── tables/                      # CSV + PNG tables
├── models/                      # joblib model bundles
└── paper/                       # final manuscript: New_Paper_WIECON_2026.docx/.pdf,
                                 # readable.md, references.bib/.xlsx, IEEE template
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NASA .mat files (~200 MB)
bash scripts/download_nasa.sh    # or see data/nasa/README.md for manual steps

# 3. Run the full pipeline (one command)
cd src && python3 generate_all_synthetic.py && \
            python3 train_full_model.py && \
            python3 run_robustness.py && \
            python3 run_sample_sweep.py && \
            python3 train_domain_rand.py && \
            python3 run_domain_rand.py && \
            python3 run_vmin_ablation.py && \
            python3 wilcoxon_test.py && \
            python3 plot_robustness.py && \
            python3 plot_sample_sweep.py && \
            python3 plot_dr_comparison.py && \
            python3 plot_distributions.py && \
            python3 plot_shap.py && \
            python3 build_robustness_tables.py

# 4. Rebuild the manuscript (optional)
python3 ../scripts/build_new_paper.py    # writes paper/New_Paper_WIECON_2026.docx
soffice --headless --convert-to pdf --outdir paper paper/New_Paper_WIECON_2026.docx
```

## Citation

If you use this code or build on the findings, please cite:

```bibtex
@inproceedings{shikdar2026robustness,
  title={How Far Does the Model Hold? Robustness of Multihorizon Battery
         Failure Intelligence Under Realistic Operating Conditions},
  author={Shikdar, T. A. and Laaksonen, H.},
  booktitle={IEEE WIECON},
  year={2026}
}
```

## License

Code: MIT. Data: NASA Ames Prognostics Data Repository (see [35] in the paper).
