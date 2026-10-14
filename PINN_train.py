"""
PINN Training — flow-map surrogate of the HORUS-2B dynamics
────────────────────────────────────────────────────────────
Trains the FlowMapMLP (256 wide, 6 residual blocks, ~802k parameters) that maps
the vehicle state at time t to the state increment over one integrator step
(dt = 0.5 s):   13 inputs  →  10 delta outputs
                (Δr, Δlat, Δlon, Δχ, Δγ, ΔCD, ΔCL, ΔV, Δq̇, Δn)

Optimised objective: data MSE (L = L_data).
The physics residual (8 EoM terms) is evaluated each epoch as a diagnostic
of physical consistency but does not enter the gradient.

Training setup
──────────────
 • Data: PINN_Tensors (split BY TRAJECTORY — no timestep leakage between
   train/val/test), scalers fit on the train split only (PINN_Scalers).
 • Mini-batching (8192), Adam, ReduceLROnPlateau, gradient clipping.
 • Best-validation checkpointing + early stopping; fully resumable.

Saves everything to:  PINN_Model/FlowMap_h256_d6/
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CLI ──────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument('--epochs', type=int, default=250, help='max training epochs')
_args = _p.parse_args()

# UTF-8 safe stdout (Windows cp1252 otherwise chokes on non-ascii)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from model import FlowMapMLP, TorchMinMaxScaler

# Use all CPU cores for BLAS / intra-op parallelism (no GPU on this machine)
torch.set_num_threads(os.cpu_count() or 8)

# ─── Paths ────────────────────────────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
tensors_path = os.path.join(current_dir, 'PINN_Tensors')
scalers_path = os.path.join(current_dir, 'PINN_Scalers')
save_dir     = os.path.join(current_dir, 'PINN_Model', 'FlowMap_h256_d6')
os.makedirs(save_dir, exist_ok=True)
ckpt_path = os.path.join(save_dir, 'checkpoint.pt')

# ─── Hyperparameters ──────────────────────────────────────────────────────────
IN_DIM, OUT_DIM = 13, 10
HIDDEN, DEPTH   = 256, 6
DROPOUT         = 0.05
BATCH_SIZE      = 8192        # large batch -> better CPU BLAS efficiency
LR              = 5e-3
EPOCHS          = _args.epochs
EARLY_STOP_PAT  = 30          # stop after this many epochs with no val improvement
CHECKPOINT_EVERY = 10



OUTPUT_NAMES = ['r','latitude','longitude','heading','flight_path',
                'CD','CL','airspeed','q_dot','g_load']

# ─── HORUS-2B / physics constants ─────────────────────────────────────────────
HORUS_MASS = 26029.0        # kg
HORUS_AREA = 110.0          # m²
HORUS_RN   = 0.8            # m, nose radius
MU         = 3.986004418e14 # m³/s²
RE         = 6_371.0e3      # m
J2         = 0.00108263
OMEGA      = 7.292115e-5    # rad/s

# Heat flux (Chapman-type stagnation-point relation)
C_STAR = 1.1097e8
VC     = 7.9054e3           # m/s, circular velocity
RHO_0  = 1.225              # kg/m³
N_HF   = 0.5
M_HF   = 3.15

# ─── Column indices ──────────────────────────────────────────────────────────
# Input  (absolute state at time t)
IDX = {
    'delta_t': 0, 'r': 1, 'mach': 2, 'latitude': 3, 'longitude': 4,
    'heading': 5, 'flight_path': 6, 'aoa': 7, 'bank': 8, 'density': 9,
    'CD': 10, 'CL': 11, 'airspeed': 12,
}
# Output (deltas over one step)
OIDX = {
    'r': 0, 'latitude': 1, 'longitude': 2, 'heading': 3, 'flight_path': 4,
    'CD': 5, 'CL': 6, 'airspeed': 7, 'q_dot': 8, 'g_load': 9,
}

# ─── Load data ────────────────────────────────────────────────────────────────
X_train, y_train = torch.load(os.path.join(tensors_path, 'train_data.pt'), weights_only=True)
X_val,   y_val   = torch.load(os.path.join(tensors_path, 'val_data.pt'),   weights_only=True)
X_test,  y_test  = torch.load(os.path.join(tensors_path, 'test_data.pt'),  weights_only=True)

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# ─── Differentiable scalers (same normalisation as preprocessing) ─────────────
torch_in_scalers = nn.ModuleDict({
    f: TorchMinMaxScaler(joblib.load(os.path.join(scalers_path, f'{f}_scaler.pkl')))
    for f in IDX
})
torch_delta_scalers = nn.ModuleDict({
    f: TorchMinMaxScaler(joblib.load(os.path.join(scalers_path, f'{f}_delta_scaler.pkl')))
    for f in OIDX
})


# ─── Physics-Informed Loss (delta formulation) ───────────────────────────────
class PINNLoss(nn.Module):
    """
    Combined data + physics loss where every output is a DELTA over one step.

    Physics residuals (8 of the 10 outputs):
        6 EoM          →  Δr, Δlat, Δlon, ΔV, Δγ, Δχ   (rotating oblate Earth, J2)
        Heat flux      →  Δq̇  =  q̇(V + ΔV, ρ) − q̇(V, ρ)
        G-load         →  Δn  =  n(next state) − n(current state)
    No physics equation exists for ΔCD, ΔCL (aero-database quantities)
    → data loss only for those two outputs.
    """

    def __init__(self, ts_in, ts_delta, criterion):
        super().__init__()
        self.ts_in     = ts_in       # input scalers   (ModuleDict)
        self.ts_delta  = ts_delta    # delta scalers   (ModuleDict)
        self.criterion = criterion   # nn.MSELoss()

    def _inv_in(self, X, feat):
        """Inverse-transform an input column → physical units."""
        return self.ts_in[feat].inverse_transform(X[:, IDX[feat]])

    def _norm_delta(self, val, feat):
        """Forward-transform a physical delta → normalised output space."""
        return self.ts_delta[feat].transform(val)

    def forward(self, X, y_pred, y_true):
        # ── Data loss ─────────────────────────────────────────────────────────
        data_loss = self.criterion(y_pred, y_true)

        # ── Decode inputs to physical units ───────────────────────────────────
        dt    = self._inv_in(X, 'delta_t')
        r     = self._inv_in(X, 'r')
        lat   = self._inv_in(X, 'latitude')
        chi   = self._inv_in(X, 'heading')
        gamma = self._inv_in(X, 'flight_path')
        V     = self._inv_in(X, 'airspeed')
        rho   = self._inv_in(X, 'density')
        CD    = self._inv_in(X, 'CD')
        CL    = self._inv_in(X, 'CL')
        bank  = self._inv_in(X, 'bank')

        # ── Aerodynamic forces ────────────────────────────────────────────────
        qS = 0.5 * rho * V**2 * HORUS_AREA
        L  = qS * CL
        D  = qS * CD

        # ── J2 gravity ────────────────────────────────────────────────────────
        g_r   = -(MU / r**2) * (
                    1 - 1.5 * J2 * (RE / r)**2 * (3 * torch.sin(lat)**2 - 1))
        g_lat = -3 * (MU / r**2) * (RE / r)**2 * J2 \
                * torch.sin(lat) * torch.cos(lat)
        g_norm = torch.sqrt(g_r**2 + g_lat**2)

        # ── EoM deltas (one explicit step of duration dt) ─────────────────────
        dr   = dt * V * torch.sin(gamma)
        dlon = dt * (V * torch.cos(gamma) * torch.sin(chi)) / (r * torch.cos(lat))
        dlat = dt * (V * torch.cos(gamma) * torch.cos(chi)) / r

        dV_dt = (
            -D / HORUS_MASS
            + g_r  * torch.sin(gamma)
            + g_lat * torch.sin(chi) * torch.cos(gamma)
            + OMEGA**2 * r * torch.cos(lat) * (
                  torch.sin(gamma) * torch.cos(lat)
                - torch.cos(gamma) * torch.sin(lat) * torch.cos(chi))
        )
        dV = dt * dV_dt

        dgamma_dt = (
            (L * torch.cos(bank)) / (HORUS_MASS * V)
            + (V / r) * torch.cos(gamma)
            + g_r  * torch.cos(gamma) / V
            - g_lat * torch.sin(chi) * torch.sin(gamma) / V
            + 2 * OMEGA * torch.cos(lat) * torch.sin(chi)
            + (OMEGA**2 * r / V) * torch.cos(lat) * (
                  torch.cos(lat) * torch.cos(gamma)
                + torch.sin(gamma) * torch.sin(lat) * torch.cos(chi))
        )
        dgamma = dt * dgamma_dt

        dchi_dt = (
            (L * torch.sin(bank)) / (HORUS_MASS * V * torch.cos(gamma))
            + (V / r) * torch.cos(gamma) * torch.tan(lat) * torch.sin(chi)
            - g_lat * torch.sin(chi) / (V * torch.cos(lat))
            + 2 * OMEGA * (
                  torch.sin(lat)
                - torch.cos(lat) * torch.cos(chi) * torch.tan(gamma))
            + (OMEGA**2 * r / (V * torch.cos(gamma)))
                * torch.cos(lat) * torch.sin(lat) * torch.sin(chi)
        )
        dchi = dt * dchi_dt

        # ── Δq̇ (heat-flux delta) ─────────────────────────────────────────────
        q_now  = (C_STAR / HORUS_RN**N_HF
                  * (rho / RHO_0)**(1.0 - N_HF) * (V / VC)**M_HF) / 1000.0
        V_next = V + dV
        q_next = (C_STAR / HORUS_RN**N_HF
                  * (rho / RHO_0)**(1.0 - N_HF) * (V_next / VC)**M_HF) / 1000.0
        dq = q_next - q_now

        # ── Δn (aerodynamic load-factor delta) ───────────────────────────────
        n_now = torch.sqrt(L**2 + D**2) / (HORUS_MASS * g_norm)

        r_next   = r   + dr
        lat_next = lat + dlat
        qS_next  = 0.5 * rho * V_next**2 * HORUS_AREA
        L_next   = qS_next * CL
        D_next   = qS_next * CD

        g_r_next   = -(MU / r_next**2) * (
                        1 - 1.5 * J2 * (RE / r_next)**2
                        * (3 * torch.sin(lat_next)**2 - 1))
        g_lat_next = -3 * (MU / r_next**2) * (RE / r_next)**2 * J2 \
                     * torch.sin(lat_next) * torch.cos(lat_next)
        g_norm_next = torch.sqrt(g_r_next**2 + g_lat_next**2)

        n_next = torch.sqrt(L_next**2 + D_next**2) / (HORUS_MASS * g_norm_next)
        dn = n_next - n_now

        # ── Normalise physics deltas into the output scaled space ─────────────
        dr_n     = self._norm_delta(dr,     'r')
        dlat_n   = self._norm_delta(dlat,   'latitude')
        dlon_n   = self._norm_delta(dlon,   'longitude')
        dchi_n   = self._norm_delta(dchi,   'heading')
        dgamma_n = self._norm_delta(dgamma, 'flight_path')
        dV_n     = self._norm_delta(dV,     'airspeed')
        dq_n     = self._norm_delta(dq,     'q_dot')
        dn_n     = self._norm_delta(dn,     'g_load')

        # ── Physics loss (mean over the 8 residuals, same scale as data loss) ─
        physics_loss = (
            self.criterion(dr_n,     y_pred[:, OIDX['r']])           +
            self.criterion(dlat_n,   y_pred[:, OIDX['latitude']])    +
            self.criterion(dlon_n,   y_pred[:, OIDX['longitude']])   +
            self.criterion(dchi_n,   y_pred[:, OIDX['heading']])     +
            self.criterion(dgamma_n, y_pred[:, OIDX['flight_path']]) +
            self.criterion(dV_n,     y_pred[:, OIDX['airspeed']])    +
            self.criterion(dq_n,     y_pred[:, OIDX['q_dot']])       +
            self.criterion(dn_n,     y_pred[:, OIDX['g_load']])
        ) / 8.0

        return data_loss, data_loss, physics_loss


# ─── Device / model / optimiser ───────────────────────────────────────────────
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('mps') if torch.backends.mps.is_available()
          else torch.device('cpu'))
print(f"Using device: {device}", flush=True)

X_val,  y_val  = X_val.to(device),  y_val.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

model     = FlowMapMLP(IN_DIM, OUT_DIM, HIDDEN, DEPTH, DROPOUT).to(device)
criterion = nn.MSELoss()
loss_fn   = PINNLoss(torch_in_scalers, torch_delta_scalers, criterion).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-5)

print(model, flush=True)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}\n", flush=True)

# ─── Resume if checkpoint exists ──────────────────────────────────────────────
start_epoch = 1
hist_train, hist_phys, hist_val = [], [], []
best_val   = float('inf')
best_epoch = 0
epochs_no_improve = 0

if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    start_epoch = ck['epoch'] + 1
    hist_train  = ck['hist_train']
    hist_phys   = ck.get('hist_phys', [])
    hist_val    = ck['hist_val']
    best_val    = ck['best_val']
    best_epoch  = ck['best_epoch']
    epochs_no_improve = ck['epochs_no_improve']
    print(f"Resumed from checkpoint at epoch {ck['epoch']} (best_val={best_val:.6e})", flush=True)

# ─── Training loop ────────────────────────────────────────────────────────────
t0 = time.time()
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    ep_total = ep_phys = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = model(Xb)
        total, dl, pl = loss_fn(Xb, y_pred, yb)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ep_total += dl.item()
        ep_phys  += pl.item()
    train_mse = ep_total / len(train_loader)
    hist_train.append(train_mse)
    hist_phys.append(ep_phys / len(train_loader))

    model.eval()
    with torch.no_grad():
        val_mse = criterion(model(X_val), y_val).item()
    hist_val.append(val_mse)
    scheduler.step(val_mse)

    # ── Best-val checkpoint (THE model we keep) ──────────────────────────────
    improved = val_mse < best_val - 1e-9
    if improved:
        best_val, best_epoch, epochs_no_improve = val_mse, epoch, 0
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict_best.pt'))
    else:
        epochs_no_improve += 1

    if epoch % 5 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        el = time.time() - t0
        print(f"Ep {epoch:4d}/{EPOCHS} | "
              f"Data={train_mse:.6e} Phys(diag)={hist_phys[-1]:.6e} | Val={val_mse:.6e} "
              f"| best={best_val:.6e}@{best_epoch} | LR={lr_now:.1e} "
              f"| {el/epoch:.1f}s/ep | no_improve={epochs_no_improve}", flush=True)

    if epoch % CHECKPOINT_EVERY == 0:
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'hist_train': hist_train, 'hist_phys': hist_phys, 'hist_val': hist_val,
                    'best_val': best_val, 'best_epoch': best_epoch,
                    'epochs_no_improve': epochs_no_improve}, ckpt_path)

    if epochs_no_improve >= EARLY_STOP_PAT:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(no val improvement for {EARLY_STOP_PAT} epochs).", flush=True)
        break

# ─── Reload BEST weights for final evaluation / saving ────────────────────────
model.load_state_dict(torch.load(os.path.join(save_dir, 'model_state_dict_best.pt'),
                                 map_location=device, weights_only=True))
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_mse  = criterion(test_pred, y_test).item()
    per_out = ((test_pred - y_test)**2).mean(dim=0).cpu().numpy()

print(f"\n=== BEST model (epoch {best_epoch}) ===", flush=True)
print(f"Best Val MSE : {best_val:.6e}", flush=True)
print(f"Test MSE     : {test_mse:.6e}", flush=True)
print("Per-output test MSE (normalised):", flush=True)
for name, v in zip(OUTPUT_NAMES, per_out):
    print(f"   {name:12s} {v:.3e}", flush=True)

# ─── Persist final artefacts ──────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict.pt'))
joblib.dump(model, os.path.join(save_dir, 'model.pkl'))
np.savetxt(os.path.join(save_dir, 'train_loss.csv'), hist_train, delimiter=',')
np.savetxt(os.path.join(save_dir, 'phys_loss.csv'),  hist_phys,  delimiter=',')
np.savetxt(os.path.join(save_dir, 'val_loss.csv'),   hist_val,   delimiter=',')
with open(os.path.join(save_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"best_epoch={best_epoch}\nbest_val_mse={best_val:.8e}\ntest_mse={test_mse:.8e}\n")
    f.write("per_output_test_mse:\n")
    for name, v in zip(OUTPUT_NAMES, per_out):
        f.write(f"  {name}={v:.6e}\n")

# ─── Loss curve ───────────────────────────────────────────────────────────────
ep_arr = np.arange(1, len(hist_train) + 1)
plt.figure(figsize=(7, 4))
plt.semilogy(ep_arr, hist_train, label='Train (data)', lw=1.2)
plt.semilogy(ep_arr, hist_val,   label='Val',          lw=1.2)
plt.axvline(best_epoch, color='green', ls='--', lw=0.8, label=f'best@{best_epoch}')
plt.xlabel('Epoch'); plt.ylabel('MSE (log)'); plt.legend()
plt.title('PINN FlowMap h256 d6')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150)

print(f"\nAll artefacts saved to:\n  {save_dir}", flush=True)
