import os
import numpy as np
import pandas as pd
import scipy.io as sio

SRC = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SRC, ".."))
DATA_PATH = os.path.join(BASE, "battery_sens", "battery_sens", "data", "nasa")
OUT_DIR = os.path.join(BASE, "data", "synthetic")
CLEAN_CSV = os.path.join(BASE, "battery_sens", "battery_sens", "data", "nasa_clean_filtered.csv")

SEVERITY_CONFIG = {
    1: {"dod_range": (0.75, 1.00), "temp_noise_std": 0.5, "rest_noise_std": 0.01},
    2: {"dod_range": (0.55, 0.75), "temp_noise_std": 1.0, "rest_noise_std": 0.02},
    3: {"dod_range": (0.35, 0.55), "temp_noise_std": 2.0, "rest_noise_std": 0.03},
    4: {"dod_range": (0.15, 0.35), "temp_noise_std": 3.0, "rest_noise_std": 0.05},
}
FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
DEFAULT_SEED = 42


def _get_battery_struct(mat):
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if hasattr(v, "cycle"):
            return v
    raise KeyError("No battery struct with .cycle found")


def _to_str(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode(errors="ignore")
    return str(x)


def _to_float_array(x):
    a = np.asarray(x)
    if a.size == 0:
        return np.array([], dtype=float)
    return a.astype(float).squeeze()


def _capacity_scalar(cap):
    cap_arr = _to_float_array(cap)
    if cap_arr.size == 0:
        return None
    if cap_arr.ndim == 0:
        val = float(cap_arr)
    else:
        val = float(cap_arr[-1])
    if not np.isfinite(val) or val <= 0:
        return None
    return val


def _compute_cumulative_ah(time, current):
    dt = np.diff(time, prepend=0)
    return np.cumsum(np.abs(current) * dt) / 3600.0


def _truncate_at_dod(voltage, current, temperature, time, capacity, dod_frac):
    ah = _compute_cumulative_ah(time, current)
    target_ah = capacity * dod_frac
    if ah[-1] <= target_ah:
        idx = len(ah)
    else:
        idx = int(np.searchsorted(ah, target_ah) + 1)
        idx = min(idx, len(ah))
    if idx < 2:
        idx = 2
    return voltage[:idx], current[:idx], temperature[:idx], time[:idx]


def _perturb_raw_cycle(voltage, current, temperature, time, capacity,
                       dod_frac, temp_noise_std, rest_noise_std, rng):
    v, i, t, tm = _truncate_at_dod(voltage, current, temperature, time, capacity, dod_frac)
    avg_v = float(np.nanmean(v))
    min_v = float(np.nanmin(v))
    avg_i = float(np.nanmean(i))
    duration = float(tm[-1] - tm[0]) if len(tm) > 1 else np.nan
    t_noisy = t + rng.normal(0, temp_noise_std, size=len(t))
    avg_t = float(np.nanmean(t_noisy))
    return {"avg_voltage": avg_v, "min_voltage": min_v,
            "avg_current": avg_i, "avg_temp": avg_t, "duration": duration}


def _build_raw_index():
    """Build {(cell, cycle): {voltage, current, ...}} from all .mat files.
    Cycle numbering matches loader.py (counts all cycles, extracts only discharge).
    """
    index = {}
    for root, _, files in os.walk(DATA_PATH):
        for f in files:
            if not f.lower().endswith(".mat"):
                continue
            path = os.path.join(root, f)
            try:
                mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
                battery = _get_battery_struct(mat)
            except Exception:
                continue
            folder = os.path.basename(root)
            cellname = os.path.splitext(f)[0]
            cell_id = f"{folder}_{cellname}"
            cyc_counter = 0
            for cyc in np.atleast_1d(battery.cycle):
                cyc_counter += 1
                ctype = _to_str(getattr(cyc, "type", None)).lower()
                if ctype != "discharge":
                    continue
                data = getattr(cyc, "data", None)
                if data is None:
                    continue
                cap = _capacity_scalar(getattr(data, "Capacity", None))
                v = _to_float_array(getattr(data, "Voltage_measured", None))
                i = _to_float_array(getattr(data, "Current_measured", None))
                t = _to_float_array(getattr(data, "Temperature_measured", None))
                tm = _to_float_array(getattr(data, "Time", None))
                if cap is None or v.size == 0 or i.size == 0 or t.size == 0 or tm.size == 0:
                    continue
                index[(cell_id, cyc_counter)] = {
                    "voltage": v, "current": i, "temperature": t, "time": tm, "capacity": cap
                }
    return index


def generate_synthetic_dataset(severity, seed=DEFAULT_SEED):
    """Generate perturbed features for exact (cell, cycle) pairs in clean data."""
    cfg = SEVERITY_CONFIG[severity]
    dod_lo, dod_hi = cfg["dod_range"]
    temp_noise_std = cfg["temp_noise_std"]
    rest_noise_std = cfg["rest_noise_std"]
    rng = np.random.default_rng(seed)

    clean = pd.read_csv(CLEAN_CSV)
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)

    raw_idx = _build_raw_index()
    print(f"  Built raw index: {len(raw_idx)} (cell, cycle) pairs")

    new_rows = []
    missing = 0
    for _, row in clean.iterrows():
        key = (row["cell"], int(row["cycle"]))
        raw = raw_idx.get(key)
        if raw is None:
            missing += 1
            new_rows.append(row.to_dict())
            continue
        dod = rng.uniform(dod_lo, dod_hi)
        pert = _perturb_raw_cycle(
            raw["voltage"], raw["current"], raw["temperature"], raw["time"],
            raw["capacity"], dod, temp_noise_std, rest_noise_std, rng
        )
        new_row = row.to_dict()
        new_row.update(pert)
        new_rows.append(new_row)

    df = pd.DataFrame(new_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"nasa_perturbed_s{severity}_s{seed}.csv")
    df.to_csv(out_path, index=False)
    matched = len(clean) - missing
    print(f"  Saved: {out_path} ({len(df)} rows, {matched}/{len(clean)} matched, {missing} missing)")
    return out_path


if __name__ == "__main__":
    for s in [1, 2, 3, 4]:
        generate_synthetic_dataset(s, seed=DEFAULT_SEED)
    print("Done.")
