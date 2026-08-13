"""SHAP summary (clean vs perturbed) + V_min interaction analysis."""
import os, sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from plot_style import apply_style
apply_style()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
FIGS_DIR = os.path.join(BASE, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
FEATURE_LABELS = ["Cycle", "V_avg", "V_min", "I_avg", "T_avg", "Duration", "SOH"]
H = 20
N_SAMPLES = 300

model_bundle = joblib.load(os.path.join(MODELS_DIR, "full_model.joblib"))
model = model_bundle["models"][H]

clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
X_clean = clean[FEATURES].values.astype(np.float64)

pert = pd.read_csv(os.path.join(DATA_DIR, "synthetic", "nasa_perturbed_s1_s42.csv"))
X_pert = pert[FEATURES].values.astype(np.float64)

explainer = shap.TreeExplainer(model)
shap_clean = explainer.shap_values(X_clean[:N_SAMPLES])
shap_pert = explainer.shap_values(X_pert[:N_SAMPLES])

# ---- Row 1: summary plots (clean vs perturbed) ----
figs = []
for vals, X_sub, title in [
    (shap_clean, X_clean[:N_SAMPLES], "Clean laboratory data"),
    (shap_pert, X_pert[:N_SAMPLES], "Perturbed (Severity 1)"),
]:
    shap.summary_plot(vals, X_sub, feature_names=FEATURE_LABELS,
                      show=False, plot_size=None, max_display=7, color_bar=True)
    f = plt.gcf()
    f.suptitle(title, fontsize=18, fontweight="bold", y=0.99)
    tmp = os.path.join(FIGS_DIR, "_shap_tmp.png")
    f.savefig(tmp, bbox_inches="tight")
    plt.close(f)
    figs.append(Image.open(tmp).convert("RGB"))
    os.remove(tmp)

w = figs[0].width + figs[1].width
h = max(figs[0].height, figs[1].height)
out_img = Image.new("RGB", (w, h), (255, 255, 255))
out_img.paste(figs[0], (0, 0))
out_img.paste(figs[1], (figs[0].width, 0))
out_path = os.path.join(FIGS_DIR, "F_SHAP_Vmin.png")
out_img.save(out_path)
print(f"Saved: {out_path}")

# ---- Row 2: V_min interaction magnitudes (clean vs perturbed) ----
inter_clean = explainer.shap_interaction_values(X_clean[:N_SAMPLES])
inter_pert = explainer.shap_interaction_values(X_pert[:N_SAMPLES])
vmin_idx = FEATURES.index("min_voltage")

fig, ax = plt.subplots(figsize=(8, 5))
labels = [l for i, l in enumerate(FEATURE_LABELS) if i != vmin_idx]
mean_clean = np.abs(inter_clean[:, vmin_idx, :]).mean(axis=0)
mean_pert = np.abs(inter_pert[:, vmin_idx, :]).mean(axis=0)
mean_clean = np.delete(mean_clean, vmin_idx)
mean_pert = np.delete(mean_pert, vmin_idx)

x = np.arange(len(labels))
width = 0.38
ax.bar(x - width / 2, mean_clean, width, color="#16a34a", label="Clean")
ax.bar(x + width / 2, mean_pert, width, color="#dc2626", label="Perturbed (S1)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean |SHAP interaction| with V_min")
ax.set_title("V_min interaction with other features", fontsize=17, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out = os.path.join(FIGS_DIR, "F_SHAP_Interaction.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
