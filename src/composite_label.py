import numpy as np
import pandas as pd

VOLTAGE_SAG_FRAC = 0.94
SOH_THRESHOLD = 0.80
EARLY_CYCLES = 10


def make_composite_fail_in_H(df: pd.DataFrame, H: int) -> np.ndarray:
    df = df.sort_values(["cell", "cycle"]).copy()
    y = pd.Series(0, index=df.index, dtype=int)

    voltage_col = "min_voltage" if "min_voltage" in df.columns else "avg_voltage"

    for _, g in df.groupby("cell", sort=False):
        g = g.sort_values("cycle").copy()

        soh_fail = g[g["SOH"] <= SOH_THRESHOLD]
        soh_fail_cycle = int(soh_fail["cycle"].iloc[0]) if len(soh_fail) > 0 else np.inf

        early = g.head(EARLY_CYCLES)
        if len(early) > 0 and early[voltage_col].notna().any():
            v_baseline = early[voltage_col].mean()
            v_threshold = v_baseline * VOLTAGE_SAG_FRAC
            v_fail = g[g[voltage_col] < v_threshold]
            v_fail_cycle = int(v_fail["cycle"].iloc[0]) if len(v_fail) > 0 else np.inf
        else:
            v_fail_cycle = np.inf

        composite_fail_cycle = min(soh_fail_cycle, v_fail_cycle)
        if composite_fail_cycle == np.inf:
            continue

        mask = (g["cycle"] + H >= composite_fail_cycle)
        y.loc[g.index] = mask.astype(int).values

    return y.loc[df.index].to_numpy()
