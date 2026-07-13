import os, sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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

try:
    explainer = shap.TreeExplainer(model)
    shap_clean = explainer.shap_values(X_clean[:500])
    shap_pert = explainer.shap_values(X_pert[:500])
    print("Using TreeExplainer")
except Exception as e:
    print(f"TreeExplainer failed ({e}), falling back to KernelExplainer")
    X_small = np.vstack([X_clean[:100], X_pert[:100]])
    explainer = shap.KernelExplainer(model.predict_proba, X_small, nsamples=100)
    shap_clean = explainer.shap_values(X_clean[:200])[1]
    shap_pert = explainer.shap_values(X_pert[:200])[1]

# render each panel as its own figure, then stitch
figs = []
for vals, X_sub, title in [
    (shap_clean, X_clean[:500], "Clean laboratory data"),
    (shap_pert, X_pert[:500], "Perturbed (Severity 1)"),
]:
    shap.summary_plot(vals, X_sub, feature_names=FEATURE_LABELS,
                      show=False, plot_size=None, max_display=7, color_bar=True)
    f = plt.gcf()
    f.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    tmp = os.path.join(FIGS_DIR, "_shap_tmp.png")
    f.savefig(tmp, dpi=200, bbox_inches="tight")
    plt.close(f)
    figs.append(Image.open(tmp).convert("RGB"))
    os.remove(tmp)

# stitch side-by-side
w = figs[0].width + figs[1].width
h = max(figs[0].height, figs[1].height)
out_img = Image.new("RGB", (w, h), (255, 255, 255))
out_img.paste(figs[0], (0, 0))
out_img.paste(figs[1], (figs[0].width, 0))
out_path = os.path.join(FIGS_DIR, "F_SHAP_Vmin.png")
out_img.save(out_path)
print(f"Saved: {out_path}")
