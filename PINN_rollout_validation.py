"""
PINN — Autoregressive ROLLOUT test (the real test for a fast integrator)
───────────────────────────────────────────────────────────────────────────
Starts from the true initial state of a HELD-OUT trajectory and integrates the
whole trajectory with the NN flow-map, feeding back its own predictions.

 • Controls (AoA, bank): TRUE recorded sequence  -> isolates integrator error
   from guidance error.
 • Density / Mach: not predicted by the net -> recomputed each step from
   r -> density and r -> speed_of_sound lookups built from the simulation data
   (same US76 atmosphere everywhere), so the rollout is self-consistent.
 • Constraints (q_dot, g_load): computed ANALYTICALLY from the rolled-out state
   (the deployment-correct, self-consistent way), compared to truth and to the
   530 kW/m^2 / 3-g limits.

Outputs per run: time histories (altitude, V, q_dot, g_load) true vs rollout,
plus a summary table of PEAK errors and constraint-violation agreement.
Saved to:  PINN_Model/FlowMap_h256_d6/rollout/
"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # avoid OMP #15 crash
import numpy as np, pandas as pd, torch, joblib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
from model import FlowMapMLP

# ─── Constants (HORUS-2B) ─────────────────────────────────────────────────────
HORUS_RN, C_STAR, VC, N_HF, M_HF, RHO_0 = 0.8, 1.1097e8, 7.9054e3, 0.5, 3.15, 1.225
HORUS_MASS, HORUS_AREA = 26_029.0, 110.0
MU, RE, J2 = 3.986004418e14, 6_371.0e3, 0.00108263
Q_LIMIT, G_LIMIT = 530.0, 3.0          # kW/m^2 , g

def q_dot_kW(rho, V):
    return (C_STAR / HORUS_RN**N_HF * (rho/RHO_0)**(1-N_HF) * (V/VC)**M_HF) / 1000.0

def g_load(rho, V, CD, CL, r, lat):
    qS = 0.5*rho*V**2*HORUS_AREA; L = qS*CL; D = qS*CD
    g_r   = (MU/r**2)*(1 - 1.5*J2*(RE/r)**2*(3*np.sin(lat)**2 - 1))
    g_lat = 3*(MU/r**2)*(RE/r)**2*J2*np.sin(lat)*np.cos(lat)
    return np.sqrt(L**2+D**2)/(HORUS_MASS*np.sqrt(g_r**2+g_lat**2))

# ─── Paths / feature order ────────────────────────────────────────────────────
cur = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(cur, 'Training_Simulations')
model_dir  = os.path.join(cur, 'PINN_Model', 'FlowMap_h256_d6')
scal_dir   = os.path.join(cur, 'PINN_Scalers')
out_dir    = os.path.join(model_dir, 'rollout'); os.makedirs(out_dir, exist_ok=True)

IN_FEATS  = ['delta_t','r','mach','latitude','longitude','heading',
             'flight_path','aoa','bank','density','CD','CL','airspeed']
OUT_FEATS = ['r','latitude','longitude','heading','flight_path',
             'CD','CL','airspeed','q_dot','g_load']
DAT_COLS  = [0,1,4,5,6,7,8,9,12,16,18,22]
DAT_NAMES = ['time','mach','latitude','longitude','heading','flight_path',
             'aoa','bank','density','CD','CL','airspeed']

# ─── Load model + scalers ─────────────────────────────────────────────────────
device = torch.device('cpu')
model = FlowMapMLP(13,10,256,6,0.05).to(device)
model.load_state_dict(torch.load(os.path.join(model_dir,'model_state_dict_best.pt'),
                                 map_location=device))
model.eval()
in_sc  = {f: joblib.load(os.path.join(scal_dir, f'{f}_scaler.pkl'))       for f in IN_FEATS}
out_sc = {f: joblib.load(os.path.join(scal_dir, f'{f}_delta_scaler.pkl')) for f in OUT_FEATS}
in_scale = np.array([in_sc[f].scale_[0] for f in IN_FEATS])
in_min   = np.array([in_sc[f].min_[0]   for f in IN_FEATS])
out_scale= np.array([out_sc[f].scale_[0] for f in OUT_FEATS])
out_min  = np.array([out_sc[f].min_[0]   for f in OUT_FEATS])

split = joblib.load(os.path.join(cur,'PINN_Tensors','run_split.pkl'))
test_runs = split['test']

# ─── Read one run's true trajectory ───────────────────────────────────────────
def read_run(rf):
    rp = os.path.join(data_dir, rf)
    dep = pd.read_csv(os.path.join(rp,'Dependent_Variables.dat'), sep=r'\s+', header=None)
    num = pd.read_csv(os.path.join(rp,'Numerical_States.dat'),   sep=r'\s+', header=None)
    n = min(len(dep), len(num)); dep, num = dep.iloc[:n], num.iloc[:n]
    r = np.sqrt(num[1]**2+num[2]**2+num[3]**2).values
    df = dep[DAT_COLS].copy(); df.columns = DAT_NAMES; df.insert(1,'r',r)
    return df

# ─── Build r -> density / speed_of_sound lookups from training runs ────────────
print("Building atmosphere lookup from training data ...", flush=True)
rr, dd, ss = [], [], []
for rf in split['train'][:120]:
    df = read_run(rf)
    rr.append(df['r'].values); dd.append(df['density'].values)
    ss.append(df['airspeed'].values / df['mach'].values)   # speed of sound = V / Mach
rr = np.concatenate(rr); dd = np.concatenate(dd); ss = np.concatenate(ss)
order = np.argsort(rr); rr, dd, ss = rr[order], dd[order], ss[order]
log_dd = np.log(np.clip(dd, 1e-30, None))
def atmo_density(r): return np.exp(np.interp(r, rr, log_dd))   # interp in log space
def atmo_sos(r):     return np.interp(r, rr, ss)
print(f"  lookup over r in [{rr.min()/1e3:.1f}, {rr.max()/1e3:.1f}] km\n", flush=True)

# ─── Autoregressive rollout for one run ───────────────────────────────────────
@torch.no_grad()
def rollout(df):
    N = len(df)
    aoa_true, bank_true = df['aoa'].values, df['bank'].values
    dt = 0.5
    # state we integrate:  r, lat, lon, heading, gamma, CD, CL, V
    s = {k: float(df[k].iloc[0]) for k in
         ['r','latitude','longitude','heading','flight_path','CD','CL','airspeed']}
    rho  = float(df['density'].iloc[0]); mach = float(df['mach'].iloc[0])
    traj = {k: np.zeros(N) for k in ['r','latitude','airspeed','density','CD','CL','q_dot','g_load']}
    def record(i):
        traj['r'][i]=s['r']; traj['latitude'][i]=s['latitude']; traj['airspeed'][i]=s['airspeed']
        traj['density'][i]=rho; traj['CD'][i]=s['CD']; traj['CL'][i]=s['CL']
        traj['q_dot'][i]=q_dot_kW(rho,s['airspeed'])
        traj['g_load'][i]=g_load(rho,s['airspeed'],s['CD'],s['CL'],s['r'],s['latitude'])
    record(0)
    for i in range(N-1):
        x = np.array([dt, s['r'], mach, s['latitude'], s['longitude'], s['heading'],
                      s['flight_path'], aoa_true[i], bank_true[i], rho, s['CD'], s['CL'],
                      s['airspeed']])
        xn = x*in_scale + in_min                                   # normalise
        yn = model(torch.tensor(xn, dtype=torch.float32)).numpy() # predict deltas (norm)
        d  = (yn - out_min)/out_scale                              # denormalise deltas
        # integrate state (first 8 deltas; q_dot/g_load deltas ignored - computed from state)
        s['r']          += d[0]; s['latitude']   += d[1]; s['longitude'] += d[2]
        s['heading']    += d[3]; s['flight_path']+= d[4]; s['CD']        += d[5]
        s['CL']         += d[6]; s['airspeed']   += d[7]
        rho  = float(atmo_density(s['r']))
        mach = float(s['airspeed']/atmo_sos(s['r']))
        record(i+1)
    return traj

# ─── True trajectory constraint signals ───────────────────────────────────────
def true_signals(df):
    r, lat = df['r'].values, df['latitude'].values
    rho, V = df['density'].values, df['airspeed'].values
    CD, CL = df['CD'].values, df['CL'].values
    return (q_dot_kW(rho,V), g_load(rho,V,CD,CL,r,lat), r, V)

# ─── Run over a set of test runs ──────────────────────────────────────────────
N_PLOT, N_EVAL = 4, 30
rows = []
print(f"Rolling out {N_EVAL} held-out test trajectories ...\n", flush=True)
for k, rf in enumerate(test_runs[:N_EVAL]):
    df = read_run(rf)
    if len(df) < 50: continue
    pr = rollout(df)
    tq, tg, tr, tV = true_signals(df)
    t = df['time'].values
    alt_t, alt_p = (tr-RE)/1e3, (pr['r']-RE)/1e3

    # peak / violation metrics
    qt_pk, qp_pk = tq.max(), pr['q_dot'].max()
    gt_pk, gp_pk = tg.max(), pr['g_load'].max()
    rows.append(dict(run=rf, n=len(df),
        alt_rmse=np.sqrt(np.mean((alt_t-alt_p)**2)),
        V_rmse=np.sqrt(np.mean((tV-pr['airspeed'])**2)),
        q_true_pk=qt_pk, q_roll_pk=qp_pk, q_pk_err=abs(qp_pk-qt_pk),
        g_true_pk=gt_pk, g_roll_pk=gp_pk, g_pk_err=abs(gp_pk-gt_pk),
        q_viol_true=qt_pk>Q_LIMIT, q_viol_roll=qp_pk>Q_LIMIT,
        g_viol_true=gt_pk>G_LIMIT, g_viol_roll=gp_pk>G_LIMIT))

    if k < N_PLOT:
        fig, ax = plt.subplots(2,2, figsize=(12,7))
        ax[0,0].plot(t, alt_t, label='true'); ax[0,0].plot(t, alt_p,'--', label='rollout')
        ax[0,0].set(title='Altitude [km]', xlabel='t [s]'); ax[0,0].legend()
        ax[0,1].plot(t, tV/1e3); ax[0,1].plot(t, pr['airspeed']/1e3,'--')
        ax[0,1].set(title='Velocity [km/s]', xlabel='t [s]')
        ax[1,0].plot(t, tq, label='true'); ax[1,0].plot(t, pr['q_dot'],'--', label='rollout')
        ax[1,0].axhline(Q_LIMIT, color='r', ls=':', label='530 kW/m²')
        ax[1,0].set(title='Heat flux q̇ [kW/m²]', xlabel='t [s]'); ax[1,0].legend()
        ax[1,1].plot(t, tg, label='true'); ax[1,1].plot(t, pr['g_load'],'--', label='rollout')
        ax[1,1].axhline(G_LIMIT, color='r', ls=':', label='3 g')
        ax[1,1].set(title='g-load [g]', xlabel='t [s]'); ax[1,1].legend()
        fig.suptitle(f'{rf}  —  full autoregressive rollout ({len(df)} steps)')
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f'rollout_{rf}.png'), dpi=130)
        plt.close(fig)

# ─── Summary ──────────────────────────────────────────────────────────────────
res = pd.DataFrame(rows)
res.to_csv(os.path.join(out_dir,'rollout_summary.csv'), index=False)
pd.set_option('display.width', 200, 'display.max_columns', 30)
print("Per-run rollout metrics (analytic constraints from rolled-out state):\n")
print(res[['run','n','alt_rmse','V_rmse','q_true_pk','q_roll_pk','q_pk_err',
           'g_true_pk','g_roll_pk','g_pk_err']].to_string(index=False,
           float_format=lambda v: f'{v:.3f}'))
# constraint-flag agreement
qa = (res.q_viol_true==res.q_viol_roll).mean()*100
ga = (res.g_viol_true==res.g_viol_roll).mean()*100
print(f"\nConstraint-violation flag agreement:  q_dot {qa:.0f}%   g_load {ga:.0f}%")
print(f"Median altitude RMSE: {res.alt_rmse.median():.3f} km   "
      f"Median V RMSE: {res.V_rmse.median():.1f} m/s")
print(f"Median peak-q̇ error: {res.q_pk_err.median():.2f} kW/m²   "
      f"Median peak-g error: {res.g_pk_err.median():.3f} g")
print(f"\nPlots ({N_PLOT}) + rollout_summary.csv saved to:\n  {out_dir}")
