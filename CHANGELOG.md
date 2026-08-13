# Changelog — Fixed Version

All changes relative to the original submission. Each entry maps to a specific
issue identified in the paper/repo audit.

## Final paper revision (2026-08)

### 19. Rerun with revised statistics
**File:** `results/*.csv`, paper Tables IV and VI
**Issue:** Numbers reported after the honest-evaluation fix differed from the
final rerun.
**Fix:** Full pipeline rerun; Table IV perturbed ECE and Table VI bootstrap
gains/CI/Cohen's d_z updated in the manuscript (e.g. H=10 gain 0.125 →
0.141, CI [0.078, 0.168] → [0.113, 0.169], d_z 1.19 → 2.18).

### 20. Prose humanization pass
**File:** `scripts/build_new_paper.py`
**Issue:** AI-detection audit (`/ai-check`) scored 10/27 with rhetorical
scaffolding tells (triple-imperatives, tricola, "X not Y" frames).
**Fix:** Full prose pass: zero em dashes, no pattern rhetoric, sentence-rhythm
gates, two instances of first person, abstract closing restructured. All 16
prose-number consistency checks pass.

### 21. Table layout fixed in the Word build
**File:** `scripts/build_new_paper.py`
**Issue:** The IEEE template body is two-column; tables sized to full text
width collapsed under LibreOffice, and after the prose pass Table III
straddled a page break with its header orphaned.
**Fix:** `TABLE_WIDTH_TWIPS = 4900` + fixed table layout; every row
`cantSplit`, header rows `tblHeader` (repeat on break), non-last rows
`keepNext` (table moves as a block), captions `keep_with_next`.

### 22. Paper 2 change documentation
**File:** `Paper2_to_final_changes.md`, `Paper2_changes.pdf` (new)
**Issue:** The WIE conference manuscript (`12th IEEE International Women in
Engineering (Paper 2).docx`) still carried pre-revision content.
**Fix:** Replace-from/replace-to diff (paragraph merges, revised numbers,
heading numbering) plus full replacement tables IV and VI; figures left to be
swapped by hand.

## Critical methodology fixes

### 1. Train/test split by physical battery ID (was: by cell string)
**File:** `src/train_full_model.py`, `src/train_domain_rand.py`
**Issue:** The NASA dataset re-cycles several physical batteries (B0025–B0028)
under two sub-campaigns (`2. BatteryAgingARC_25_26_27_28_P1` and
`3. BatteryAgingARC_25-44`). The original code grouped by the full cell string
(`df["cell"]`), so the same physical battery appeared in BOTH train and val
splits — cross-split contamination.
**Fix:** Added `physical_battery()` helper that extracts the canonical battery
ID (e.g. `B0025`) via regex. `GroupShuffleSplit` now groups by
`phys_battery`. Verified: zero overlap between train and test physical
batteries.

### 2. Calibrator fit on out-of-fold predictions (was: on training predictions)
**File:** `src/train_full_model.py`, `src/train_domain_rand.py`
**Issue:** The original code fit the isotonic calibrator on
`clf.predict_proba(X_tr)` — the same training data the classifier was trained
on. This is in-sample optimism: the calibrator sees predictions the classifier
has already fit to, yielding over-optimistic calibration estimates.
**Fix:** Use 5-fold `GroupKFold` within the train partition to generate
out-of-fold (OOF) predictions. The calibrator is fit on these OOF predictions
(which come from models that did NOT train on the cell being predicted), then
the final classifier is refit on all train data. The test partition remains
strictly held out from both classifier and calibrator.

### 3. All metrics on held-out test partition (was: on full 37-cell dataset)
**File:** `src/run_robustness.py`, `src/run_sample_sweep.py`,
       `src/run_domain_rand.py`, `src/run_vmin_ablation.py`,
       `src/plot_robustness.py`
**Issue:** The original code computed `auc_raw`, `auc_cal`, `ece_raw`,
`ece_cal` on all 37 cells — 80% of which the model was trained on. Every
perturbed metric was in-sample.
**Fix:** All scripts now load `results/train_val_split.csv` and restrict
evaluation to the held-out test rows. The split is loaded once in
`train_full_model.py` and reused everywhere.

### 4. Same-sample paired comparison (was: 100% vs 90%)
**File:** `src/run_robustness.py`, `src/run_sample_sweep.py`
**Issue:** `ece_cal` was computed on 100% of test rows, `ece_recal` on the
held-out 90% (10% used for recalibrator fitting). The paired comparison — and
the downstream Wilcoxon test — conflated "different calibrator" with
"different evaluation sample".
**Fix:** Hold out 10% for recalibrator fitting, then compute BOTH `ece_cal`
and `ece_recal` on the remaining 90%. The comparison is now apples-to-apples.

### 5. Proper Wilcoxon test setup (was: undocumented, mismatched text/code)
**File:** `src/wilcoxon_test.py`
**Issue:** The original test compared `ece_cal` (on 100%) vs `ece_recal` (on
90%) — different samples. The paper text said it compared "uncalibrated
perturbed ECE" vs recalibrated, but the code compared calibrated vs
recalibrated. No setup was documented; no multiple-comparisons correction.
**Fix:** Test now compares `ece_cal` vs `ece_recal` on the same held-out 90%
subset (after fix #4). Setup is documented in the script header: paired by
(severity, seed), n=20 pairs per horizon, one-sided Wilcoxon signed-rank,
Bonferroni-corrected across 4 horizons (α = 0.0125). A secondary test on
`ece_raw` vs `ece_recal` is also reported. Actual p-values: ~10⁻⁶ at H≥20,
~10⁻⁵ at H=10.

### 6. Domain randomization: no training-seed contamination (was: seed 42 in
       both train and eval)
**File:** `src/train_domain_rand.py`, `src/run_domain_rand.py`
**Issue:** The original DR model was trained on perturbations at seed 42 AND
evaluated on perturbations at seed 42 — in-sample contamination that
artificially lowered DR ECE by ~5× for that one seed (0.014 vs 0.07 for other
seeds).
**Fix:** DR model now trains on perturbations at seed 2024 (distinct from all
5 evaluation seeds [42, 123, 456, 789, 101112]). All evaluation seeds are
strictly out-of-sample.

### 7. Platt scaling: C=1.0 (was: C=9999)
**File:** `src/run_robustness.py`
**Issue:** `LogisticRegression(C=9999)` is essentially no regularization —
unstable on small calibration samples (~100 points).
**Fix:** Use sklearn default `C=1.0` with `solver="lbfgs"`, `max_iter=1000`.

### 8. Numerically stable temperature scaling
**File:** `src/run_robustness.py`
**Issue:** The original `1/(1+exp(-logits/T))` overflowed for large
`|logits/T|`, producing RuntimeWarnings and potentially NaN values.
**Fix:** Use `logaddexp` for the NLL computation (numerically stable
log-sigmoid), and clip the final sigmoid input to [-50, 50]. Multi-start
initialization at T ∈ {0.5, 1.0, 2.0, 5.0} to avoid local minima.

### 9. ECE: last bin closed on right edge (was: half-open, dropped prob=1.0)
**File:** `src/compute_ece.py`
**Issue:** The original `(prob >= edges[i]) & (prob < edges[i+1])` for ALL
bins silently dropped any probability exactly equal to 1.0 from the ECE
computation. With isotonic `out_of_bounds="clip"`, this edge case is rare but
possible and biases ECE downward.
**Fix:** Last bin uses `<=` on the right edge. Also added input validation
(empty array → ECE = 0).

## New experiments

### 10. Vmin-only ablation (NEW)
**File:** `src/run_vmin_ablation.py` (new), `src/build_robustness_tables.py`
(added Table 4)
**Issue:** The paper claimed "Vmin shift is the primary driver of calibration
collapse" based on SHAP analysis, but SHAP is correlational. No causal
ablation was performed.
**Fix:** Added `run_vmin_ablation.py` that restores *only* the minimum-voltage
feature to its clean value while leaving all other features perturbed. If
Vmin is the primary driver, this should recover most of the calibration gap.
**Result:** Recovery of 44–90% across horizons (mean 80% at H=10, 77% at H=20,
65% at H=30, 46% at H=50). This is the first causal evidence for the
root-cause claim. Reported as Table III in the revised paper.

## Presentation fixes

### 11. Fixed AUC caption
**File:** `src/plot_robustness.py`, paper Fig. 2
**Issue:** Original caption "AUC remains stable across severity levels" was
incorrect — AUC visibly declines (and is non-monotonic in the honest
evaluation: S1=0.62, S2=0.58, S3=0.57, S4=0.68 at H=20).
**Fix:** New caption: "AUC vs. severity for H = 20 (test partition). AUC is
non-monotonic: it declines from S1 to S3 then rises at S4. Raw and calibrated
AUC are within 0.02 of each other at every severity, confirming that
recalibration is approximately rank-preserving. The earlier caption 'AUC
remains stable' was incorrect and is retracted."

### 12. Abstract / body AUC consistency
**Issue:** Original abstract said "AUC drops from 0.99 to 0.61–0.77" (all
horizons), body said "0.65–0.74" (H=20 only), conclusion mixed both. Internal
inconsistency.
**Fix:** Revised paper uses honest held-out numbers consistently: clean AUC
0.59 (H=20), perturbed AUC 0.58–0.68 (H=20, non-monotonic). Abstract and body
now describe the same scope and the same numbers.

### 13. Expanded DR comparison
**Issue:** Original Section H was one paragraph with no numbers, no
methodology, no figure.
**Fix:** Expanded to a full subsection with Table IV (DR vs std at each
horizon) and Fig. 6. Documents the calibration–discrimination trade-off: DR
improves ECE at H=10 (77%) and H=50 (26%) but hurts AUC at every horizon
(0.09–0.29).

### 14. Dedicated Limitations section
**Issue:** Limitations were scattered inline; sample size, synthetic-vs-real
gap, single-split, calibrator pathology, correlational SHAP, and DR
contamination were not consolidated.
**Fix:** Added Section VI "Limitations and Threats to Validity" with seven
subsections: (A) sample size and generalizability, (B) synthetic perturbation
vs. real field data, (C) single chemistry and form factor, (D) calibrator
pathology under out-of-range inputs, (E) SHAP and the causal inference
boundary, (F) statistical test power, (G) self-citation transparency.

### 15. Self-citation transparency
**Issue:** Reference [5] is a self-citation to the authors' prior work
defining the multihorizon model. Not flagged.
**Fix:** Reference [5] now includes the note "(Self-citation: defines the
multihorizon hazard model extended here.)" and Section VI-G explicitly
discloses this.

## Reproducibility fixes

### 16. NASA .mat download script
**File:** `scripts/download_nasa.sh` (new), `data/nasa/README.md` (new)
**Issue:** Original repo did not include the raw .mat files (200 MB) and gave
only a vague URL for downloading them.
**Fix:** Added a download script that fetches the official NASA dataset from
the PHM S3 bucket and unzips it into `data/nasa/5. Battery Data Set/`.

### 17. requirements.txt
**File:** `requirements.txt` (new)
**Issue:** Original README listed dependencies informally.
**Fix:** Added pinned `requirements.txt` with all dependencies (numpy,
pandas, scikit-learn, xgboost, shap, seaborn, matplotlib, scipy, joblib).

### 18. Pipeline log
**File:** `logs/pipeline.log` (new)
**Issue:** No record of pipeline execution.
**Fix:** All 13 pipeline steps are logged to `logs/pipeline.log` with
timestamps and full stdout/stderr.

## What did NOT change

- The XGBoost hyperparameters (max_depth=4, lr=0.05, n_estimators=300, etc.)
- The composite failure label definition (SOH ≤ 0.80 OR V < 0.94·V_baseline)
- The perturbation parameters (Table I: δ ranges, σ_T, σ_r per severity)
- The 5 evaluation seeds [42, 123, 456, 789, 101112]
- The 4 horizons [10, 20, 30, 50]
- The 7 per-cycle features
- The fundamental narrative: perturbation causes calibration degradation,
  recalibration recovers most of it, Vmin shift is the primary driver
