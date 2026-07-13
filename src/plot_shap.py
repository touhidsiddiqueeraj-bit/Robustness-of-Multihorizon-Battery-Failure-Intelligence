import os, sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
FIGS_DIR = os.path.join(BASE, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
FEATURE_LABELS = ["Cycle", "V_avg", "V_min", "I_avg", "T_avg", "Duration", "SOH"]
H = 20

model_bundle = joblib.load(os.path.join(MODELS_DIR, "full_model.joblib"))
model = model_bundle["models"][H]

clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
X_clean = clean[FEATURES].values.astype(np.float64)

pert = pd.read_csv(os.path.join(DATA_DIR, "synthetic", "nasa_perturbed_s1_s42.csv"))
X_pert = pert[FEATURES].values.astype(np.float64)

use_kernel = False
try:
    explainer = shap.TreeExplainer(model)
    shap_clean = explainer.shap_values(X_clean[:500])
    shap_pert = explainer.shap_values(X_pert[:500])
    print("Using TreeExplainer")
except Exception as e:
    print(f"TreeExplainer failed ({e}), falling back to KernelExplainer")
    use_kernel = True
    X_small = np.vstack([X_clean[:100], X_pert[:100]])
    explainer = shap.KernelExplainer(model.predict_proba, X_small, nsamples=100)
    shap_clean = explainer.shap_values(X_clean[:200])[1]
    shap_pert = explainer.shap_values(X_pert[:200])[1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, X_sub, shap_vals, title in [
    (axes[0], X_clean[:500], shap_clean, "Clean laboratory data"),
    (axes[1], X_pert[:500], shap_pert, "Perturbed (Severity 1)"),
]:
    shap.summary_plot(shap_vals, X_sub, feature_names=FEATURE_LABELS,
                      show=False, plot_size=None, max_display=7, color_bar=True)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=11)

plt.tight_layout()
out = os.path.join(FIGS_DIR, "F_SHAP_Vmin.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# Also do a waterfall for one sample to show Vmin dominance
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.5))
for i, (X_sub, shap_vals, title) in enumerate([
    (X_clean[:500], shap_clean, "Clean — single prediction"),
    (X_pert[:500], shap_pert, "Perturbed — single prediction"),
]):
    shap.waterfall_plot(
        shap.Explanation(values=shap_vals[0], base_values=explainer.expected_shape if use_kernel else explainer.expected_value,
                         data=X_sub[0], feature_names=FEATURE_LABELS),
        show=False, max_display=7
    )
    ax2[i].set_title(title, fontsize=14, fontweight="bold")

plt.tight_layout()
out2 = os.path.join(FIGS_DIR, "F_SHAP_waterfall.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()
