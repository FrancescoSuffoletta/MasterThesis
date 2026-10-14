"""
PINN Data Preprocessing  (by-trajectory split, leak-free)
──────────────────────────────────────────────────────────
Builds the training data of the PINN flow-map surrogate from the TUDAT
re-entry simulations in Training_Simulations/run_XXXXX.

Key design choices:
  • Samples are (state at t)  →  (state increment over one 0.5 s step):
    13 inputs → 10 delta outputs (Δr, Δlat, Δlon, Δχ, Δγ, ΔCD, ΔCL, ΔV, Δq̇, Δn).
  • Split is BY TRAJECTORY (whole run_XXXXX folders), not by timestep
    -> no adjacent-step leakage between train/val/test; the validation and
       test errors honestly measure generalisation to UNSEEN trajectories
       (the actual deployment scenario of the surrogate).
  • Scalers are fit on the TRAIN split ONLY (fitting on all data would leak
    val/test statistics into the normalisation).
  • The derived constraint quantities q̇ (stagnation heat flux) and n (aero
    load factor) are computed analytically at every timestep and their deltas
    added as supervised outputs.

Persistence:  PINN_Tensors/  (train/val/test tensors + run_split.pkl)
              PINN_Scalers/  (one MinMaxScaler per input / per delta output)
"""

import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# ─── HORUS-2B constants ───────────────────────────────────────────────────────
HORUS_RN   = 0.8
C_STAR     = 1.1097e8
VC         = 7.9054e3
N_HF       = 0.5
M_HF       = 3.15
RHO_0      = 1.225
HORUS_MASS = 26_029.0
HORUS_AREA = 110.0
MU         = 3.986004418e14
RE         = 6_371.0e3
J2         = 0.00108263

def compute_q_dot_kW(density, airspeed):
    return (C_STAR / HORUS_RN**N_HF
            * (density / RHO_0)**(1.0 - N_HF)
            * (airspeed / VC)**M_HF) / 1000.0

def compute_g_load(density, airspeed, CD, CL, r, latitude):
    qS   = 0.5 * density * airspeed**2 * HORUS_AREA
    L    = qS * CL
    D    = qS * CD
    g_r   = (MU / r**2) * (1 - 1.5 * J2 * (RE / r)**2
                            * (3 * np.sin(latitude)**2 - 1))
    g_lat = 3 * (MU / r**2) * (RE / r)**2 * J2 \
            * np.sin(latitude) * np.cos(latitude)
    g_norm = np.sqrt(g_r**2 + g_lat**2)
    return np.sqrt(L**2 + D**2) / (HORUS_MASS * g_norm)

# ─── Paths ────────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(__file__)
data_dir    = os.path.join(current_dir, "Training_Simulations")

# ─── Feature definitions ──────────────────────────────────────────────────────
INPUT_FEATURES = [
    'delta_t', 'r', 'mach', 'latitude', 'longitude', 'heading',
    'flight_path', 'aoa', 'bank', 'density', 'CD', 'CL', 'airspeed'
]
OUTPUT_FEATURES = [
    'r', 'latitude', 'longitude', 'heading',
    'flight_path', 'CD', 'CL', 'airspeed', 'q_dot', 'g_load'
]
STATE_DELTA_FEATURES = [
    'r', 'latitude', 'longitude', 'heading',
    'flight_path', 'CD', 'CL', 'airspeed'
]

DAT_COLS  = [0, 1, 4, 5, 6, 7, 8, 9, 12, 16, 18, 22]
DAT_NAMES = [
    'time', 'mach', 'latitude', 'longitude', 'heading',
    'flight_path', 'aoa', 'bank', 'density', 'CD', 'CL', 'airspeed'
]

# ─── Build (input, output) pairs for ONE run ──────────────────────────────────
def build_run(run_path):
    dep_path = os.path.join(run_path, 'Dependent_Variables.dat')
    num_path = os.path.join(run_path, 'Numerical_States.dat')
    if not (os.path.exists(dep_path) and os.path.exists(num_path)):
        return None
    try:
        dep = pd.read_csv(dep_path, sep=r'\s+', header=None)
        num = pd.read_csv(num_path, sep=r'\s+', header=None)
    except Exception:
        return None

    n_rows = min(len(dep), len(num))
    dep = dep.iloc[:n_rows].reset_index(drop=True)
    num = num.iloc[:n_rows].reset_index(drop=True)
    if n_rows < 3:
        return None

    r = np.sqrt(num[1]**2 + num[2]**2 + num[3]**2)
    df = dep[DAT_COLS].copy()
    df.columns = DAT_NAMES
    df.insert(1, 'r', r.values)

    dt_arr = np.diff(df['time'].values)
    df['delta_t'] = np.append(dt_arr, dt_arr[-1])

    df['q_dot']  = compute_q_dot_kW(df['density'].values, df['airspeed'].values)
    df['g_load'] = compute_g_load(
        df['density'].values, df['airspeed'].values,
        df['CD'].values, df['CL'].values,
        df['r'].values, df['latitude'].values
    )

    input_data = df[INPUT_FEATURES].iloc[:-1].reset_index(drop=True)

    delta_data = pd.DataFrame()
    for feat in STATE_DELTA_FEATURES:
        delta_data[feat] = df[feat].iloc[1:].values - df[feat].iloc[:-1].values
    delta_data['q_dot']  = df['q_dot'].iloc[1:].values  - df['q_dot'].iloc[:-1].values
    delta_data['g_load'] = df['g_load'].iloc[1:].values - df['g_load'].iloc[:-1].values

    return input_data.values.astype(np.float32), delta_data.values.astype(np.float32)

# ─── Split RUN FOLDERS into train/val/test (NOT timesteps) ────────────────────
run_folders = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])
print(f"Found {len(run_folders)} run folders")

runs_train, runs_tmp = train_test_split(run_folders, test_size=0.30, random_state=42)
runs_val,   runs_test = train_test_split(runs_tmp,   test_size=0.50, random_state=42)
print(f"Run split  →  train {len(runs_train)}  |  val {len(runs_val)}  |  test {len(runs_test)}")

# ─── Assemble each split ──────────────────────────────────────────────────────
def assemble(run_list, label):
    ins, outs, skipped = [], [], 0
    for rf in run_list:
        res = build_run(os.path.join(data_dir, rf))
        if res is None:
            skipped += 1
            continue
        ins.append(res[0]); outs.append(res[1])
    X = np.concatenate(ins, axis=0)
    y = np.concatenate(outs, axis=0)
    print(f"  {label:5s}: {len(run_list)-skipped} runs, {len(X):,} samples ({skipped} skipped)")
    return X, y

print("\nAssembling splits:")
X_train_raw, y_train_raw = assemble(runs_train, "train")
X_val_raw,   y_val_raw   = assemble(runs_val,   "val")
X_test_raw,  y_test_raw  = assemble(runs_test,  "test")

# ─── Fit scalers on TRAIN ONLY ────────────────────────────────────────────────
input_scalers = {}
for i, feature in enumerate(INPUT_FEATURES):
    sc = MinMaxScaler(feature_range=(-1, 1))
    if feature == 'delta_t':
        sc.fit(np.array([[0.0], [10.0]]))           # fixed physical range
    else:
        sc.fit(X_train_raw[:, i].reshape(-1, 1))
    input_scalers[feature] = sc

delta_scalers = {}
for j, feature in enumerate(OUTPUT_FEATURES):
    sc = MinMaxScaler(feature_range=(-1, 1))
    sc.fit(y_train_raw[:, j].reshape(-1, 1))
    delta_scalers[feature] = sc

# ─── Persist scalers ──────────────────────────────────────────────────────────
scalers_path = os.path.join(current_dir, 'PINN_Scalers')
os.makedirs(scalers_path, exist_ok=True)
for name, sc in input_scalers.items():
    joblib.dump(sc, os.path.join(scalers_path, f'{name}_scaler.pkl'))
for name, sc in delta_scalers.items():
    joblib.dump(sc, os.path.join(scalers_path, f'{name}_delta_scaler.pkl'))
print(f"\nScalers (fit on TRAIN only) saved to {scalers_path}")

# ─── Normalise each split with the train-fit scalers ──────────────────────────
def normalise(X_raw, y_raw):
    Xn = np.column_stack([
        input_scalers[f].transform(X_raw[:, i].reshape(-1, 1)).ravel()
        for i, f in enumerate(INPUT_FEATURES)
    ]).astype(np.float32)
    yn = np.column_stack([
        delta_scalers[f].transform(y_raw[:, j].reshape(-1, 1)).ravel()
        for j, f in enumerate(OUTPUT_FEATURES)
    ]).astype(np.float32)
    return torch.from_numpy(Xn), torch.from_numpy(yn)

X_train, y_train = normalise(X_train_raw, y_train_raw)
X_val,   y_val   = normalise(X_val_raw,   y_val_raw)
X_test,  y_test  = normalise(X_test_raw,  y_test_raw)

# Note: val/test may slightly exceed [-1,1] where unseen runs exceed train range
print(f"\nNormalised ranges (train fit):")
print(f"  X_val  min/max: {float(X_val.min()):.3f} / {float(X_val.max()):.3f}")
print(f"  y_val  min/max: {float(y_val.min()):.3f} / {float(y_val.max()):.3f}")

# ─── Persist tensors ──────────────────────────────────────────────────────────
tensors_path = os.path.join(current_dir, 'PINN_Tensors')
os.makedirs(tensors_path, exist_ok=True)
torch.save((X_train, y_train), os.path.join(tensors_path, 'train_data.pt'))
torch.save((X_val,   y_val),   os.path.join(tensors_path, 'val_data.pt'))
torch.save((X_test,  y_test),  os.path.join(tensors_path, 'test_data.pt'))

# Save which runs went where (reproducibility + later trajectory testing)
joblib.dump(
    {'train': runs_train, 'val': runs_val, 'test': runs_test},
    os.path.join(tensors_path, 'run_split.pkl')
)
print(f"\nTrain: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
print(f"Tensors + run_split.pkl saved to {tensors_path}")
print("Preprocessing v3 (by-run, leak-free) complete.")
