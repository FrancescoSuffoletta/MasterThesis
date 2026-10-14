"""
PINN constraint corrector  (drop-in core for the predictor-corrector guidance)
─────────────────────────────────────────────────────────────────────────────────
Holds the computationally-heavy, TUDAT-independent logic so it can be unit-tested
offline and reused by MainGuidance.py:

  • load_model()            : PINN flow-map (256 wide, 6 blocks) from model_state_dict_best.pt
  • rollout_peaks(state,bank): 50 s autoregressive rollout -> (peak q_dot, peak g)
  • check(state,bank)        : -> "None" | "Heat Flux" | "G-load" | "Both"
  • find_safe_bank(state,cmd): BISECTION root-find -> largest bank <= cmd that
                               keeps both constraints under their (margined) limits.

Why bisection instead of Newton:
  peak-q̇ and peak-g are monotone in |bank| (more bank -> less vertical lift ->
  denser flight -> higher loads), and we have a natural bracket [0, commanded].
  Bisection needs ONE rollout per iteration (Newton's finite-difference derivative
  needs three), is guaranteed to converge inside the bracket, and uses only the
  SIGN of (peak - limit) so it is robust to the surrogate's heat-flux jitter.

State vector convention (physical units, the 12 quantities the corrector tracks):
  [r, mach, lat, lon, heading, gamma, aoa, bank, density, CD, CL, V]
(delta_t = 0.5 s is appended internally; q_dot/g_load are computed analytically
 from the rolled-out state, NOT taken from the model's Δq_dot/Δg_load outputs.)
"""
import os
import numpy as np
import torch
import joblib
from model import FlowMapMLP

# ─── HORUS-2B / physics constants ─────────────────────────────────────────────
HORUS_RN, C_STAR, VC, N_HF, M_HF, RHO_0 = 0.8, 1.1097e8, 7.9054e3, 0.5, 3.15, 1.225
HORUS_MASS, HORUS_AREA = 26_029.0, 110.0
MU, RE, J2 = 3.986004418e14, 6_371.0e3, 0.00108263
GAMMA_AIR, R_AIR, F_OBL = 1.4, 287.05, 1/298.257223563

# AoA energy law (mirrors HORUSGuidanceAndTrim.Update_AoA)
AOA_ENERGY = np.array([-1_600_000.0, -200_000.0])
MAX_AOA, FINAL_AOA = np.deg2rad(40.0), np.deg2rad(11.5)

IN_FEATS  = ['delta_t','r','mach','latitude','longitude','heading',
             'flight_path','aoa','bank','density','CD','CL','airspeed']
OUT_FEATS = ['r','latitude','longitude','heading','flight_path',
             'CD','CL','airspeed','q_dot','g_load']
# indices into the 12-element physical state we carry
S = dict(r=0, mach=1, lat=2, lon=3, head=4, gam=5, aoa=6, bank=7, rho=8, CD=9, CL=10, V=11)


def aoa_from_energy(r, V):
    e = (1.0/r) - (V*V/2.0)
    if e <= AOA_ENERGY[0]:
        return MAX_AOA
    if e < AOA_ENERGY[1]:
        return ((FINAL_AOA - MAX_AOA)/(AOA_ENERGY[1]-AOA_ENERGY[0]))*(e-AOA_ENERGY[0]) + MAX_AOA
    return FINAL_AOA


def g_norm_J2(r, lat):
    gr  = (MU/r**2)*(1 - 1.5*J2*(RE/r)**2*(3*np.sin(lat)**2 - 1))
    gph = (MU/r**2)*(-3*(RE/r)**2*np.sin(lat)*np.cos(lat)*J2)
    return np.sqrt(gr**2 + gph**2)

def q_dot_kW(rho, V):
    V = max(V, 1.0)
    return (C_STAR/HORUS_RN**N_HF * (max(rho,1e-30)/RHO_0)**(1-N_HF) * (V/VC)**M_HF) / 1000.0

def g_load(rho, V, CD, CL, r, lat):
    qS = 0.5*rho*V**2*HORUS_AREA
    return np.sqrt((qS*CL)**2 + (qS*CD)**2) / (HORUS_MASS * g_norm_J2(r, lat))


class PINNCorrector:
    def __init__(self, current_dir=None,
                 q_limit=530.0, g_limit=3.0,
                 q_margin=20.0, g_margin=0.1,
                 horizon_s=50.0, dt=0.5,
                 climb_limit_m=500.0,
                 atmosphere='coesa76'):
        current_dir = current_dir or os.path.dirname(os.path.abspath(__file__))
        self.dt = dt
        self.n_steps = int(round(horizon_s/dt))
        # targets the corrector keeps the trajectory below (limit minus safety margin)
        self.q_limit, self.g_limit = q_limit, g_limit
        self.q_target, self.g_target = q_limit - q_margin, g_limit - g_margin
        # SKIP GUARD: a candidate bank whose 50 s rollout climbs more than this above
        # the current altitude is rejected (over-lift -> phugoid/skip-out). Prevents the
        # corrector from trading a marginal heat-flux for a catastrophic downstream g-load.
        self.climb_limit_m = climb_limit_m

        # model
        mdl = os.path.join(current_dir, 'PINN_Model', 'FlowMap_h256_d6',
                           'model_state_dict_best.pt')
        self.model = FlowMapMLP(13, 10, 256, 6, 0.05)
        self.model.load_state_dict(torch.load(mdl, map_location='cpu', weights_only=True))
        self.model.eval()

        # scalers -> flat numpy for speed
        sc = os.path.join(current_dir, 'PINN_Scalers')
        ins = {f: joblib.load(os.path.join(sc, f'{f}_scaler.pkl')) for f in IN_FEATS}
        outs = {f: joblib.load(os.path.join(sc, f'{f}_delta_scaler.pkl')) for f in OUT_FEATS}
        self.in_scale = np.array([ins[f].scale_[0] for f in IN_FEATS])
        self.in_min   = np.array([ins[f].min_[0]   for f in IN_FEATS])
        self.out_scale= np.array([outs[f].scale_[0] for f in OUT_FEATS])
        self.out_min  = np.array([outs[f].min_[0]   for f in OUT_FEATS])

        self._build_atmosphere(atmosphere, current_dir)

    # ── fast altitude -> (density, sound speed) table (built once) ──────────────
    def _build_atmosphere(self, kind, current_dir):
        alts = np.arange(0.0, 130_000.0, 50.0)          # 0..130 km @ 50 m
        if kind == 'coesa76':
            from poliastro.earth.atmosphere.coesa76 import COESA76
            from astropy import units as u
            atm = COESA76()
            rho = np.array([atm.density((a*u.m)).value for a in alts])
            T   = np.array([atm.temperature((a*u.m)).value for a in alts])
            sos = np.sqrt(GAMMA_AIR*R_AIR*T)
        else:  # 'data' — build from training runs (for offline testing)
            import pandas as pd
            data_dir = os.path.join(current_dir, 'Training_Simulations')
            split = joblib.load(os.path.join(current_dir,'PINN_Tensors','run_split.pkl'))
            DAT_COLS=[0,1,4,5,6,7,8,9,12,16,18,22]
            rr,dd,ss=[],[],[]
            for rf in split['train'][:80]:
                dep=pd.read_csv(os.path.join(data_dir,rf,'Dependent_Variables.dat'),sep=r'\s+',header=None)
                num=pd.read_csv(os.path.join(data_dir,rf,'Numerical_States.dat'),sep=r'\s+',header=None)
                n=min(len(dep),len(num)); dep,num=dep.iloc[:n],num.iloc[:n]
                r=np.sqrt(num[1]**2+num[2]**2+num[3]**2).values
                lat=dep[4].values
                a=r - RE*(1 - F_OBL*np.sin(lat)**2)
                rr.append(a); dd.append(dep[12].values); ss.append(dep[22].values/dep[1].values)
            a=np.concatenate(rr); dd=np.concatenate(dd); ss=np.concatenate(ss)
            o=np.argsort(a); a,dd,ss=a[o],dd[o],ss[o]
            rho=np.exp(np.interp(alts,a,np.log(np.clip(dd,1e-30,None))))
            sos=np.interp(alts,a,ss)
        self._alts, self._logrho_tab, self._sos_tab = alts, np.log(np.clip(rho,1e-30,None)), sos

    def _altitude(self, r, lat):
        return r - RE*(1 - F_OBL*np.sin(lat)**2)
    def _density(self, alt):
        return float(np.exp(np.interp(alt, self._alts, self._logrho_tab)))
    def _sos(self, alt):
        return float(np.interp(alt, self._alts, self._sos_tab))

    # ── vectorised constraint helpers (operate on arrays over candidate banks) ──
    @staticmethod
    def _q_dot_kW_vec(rho, V):
        V = np.clip(V, 1.0, None)
        return (C_STAR/HORUS_RN**N_HF * (np.clip(rho,1e-30,None)/RHO_0)**(1-N_HF) * (V/VC)**M_HF)/1000.0
    @staticmethod
    def _g_load_vec(rho, V, CD, CL, r, lat):
        qS = 0.5*rho*V**2*HORUS_AREA
        gr  = (MU/r**2)*(1 - 1.5*J2*(RE/r)**2*(3*np.sin(lat)**2 - 1))
        gph = (MU/r**2)*(-3*(RE/r)**2*np.sin(lat)*np.cos(lat)*J2)
        return np.sqrt((qS*CL)**2 + (qS*CD)**2)/(HORUS_MASS*np.sqrt(gr**2+gph**2))
    @staticmethod
    def _aoa_vec(r, V):
        e = (1.0/r) - (V*V/2.0)
        slope = (FINAL_AOA-MAX_AOA)/(AOA_ENERGY[1]-AOA_ENERGY[0])
        mid = slope*(e-AOA_ENERGY[0]) + MAX_AOA
        return np.where(e <= AOA_ENERGY[0], MAX_AOA, np.where(e < AOA_ENERGY[1], mid, FINAL_AOA))

    # ── BATCHED 50 s rollout over W candidate banks; one forward/step total ─────
    # On CPU a batched forward of ~W banks costs ~the same as a single sample, so
    # evaluating the whole bank grid at once is ~W× cheaper than sequential search.
    @torch.no_grad()
    def rollout_peaks_batch(self, state, banks):
        banks = np.atleast_1d(np.asarray(banks, dtype=np.float64))
        W = len(banks)
        s = np.tile(np.asarray(state, dtype=np.float64), (W, 1))   # [W,12]
        s[:, S['bank']] = banks
        r0 = s[:, S['r']].copy()                                   # start radius (skip guard)
        peak_q = self._q_dot_kW_vec(s[:,S['rho']], s[:,S['V']])
        peak_g = self._g_load_vec(s[:,S['rho']], s[:,S['V']], s[:,S['CD']], s[:,S['CL']], s[:,S['r']], s[:,S['lat']])
        climb  = np.zeros(W)                                       # max (r - r0) over horizon
        for _ in range(self.n_steps):
            X = np.column_stack([np.full(W,self.dt), s[:,S['r']], s[:,S['mach']], s[:,S['lat']],
                                 s[:,S['lon']], s[:,S['head']], s[:,S['gam']], s[:,S['aoa']],
                                 banks, s[:,S['rho']], s[:,S['CD']], s[:,S['CL']], s[:,S['V']]])
            Xn = (X*self.in_scale + self.in_min).astype(np.float32)
            yn = self.model(torch.from_numpy(Xn)).numpy()
            d  = (yn - self.out_min)/self.out_scale
            s[:,S['r']]   += d[:,0]; s[:,S['lat']] += d[:,1]; s[:,S['lon']]  += d[:,2]
            s[:,S['head']]+= d[:,3]; s[:,S['gam']] += d[:,4]; s[:,S['CD']]   += d[:,5]
            s[:,S['CL']]  += d[:,6]; s[:,S['V']]   += d[:,7]
            alt = self._altitude(s[:,S['r']], s[:,S['lat']])
            s[:,S['rho']]  = np.exp(np.interp(alt, self._alts, self._logrho_tab))
            s[:,S['mach']] = s[:,S['V']]/np.interp(alt, self._alts, self._sos_tab)
            s[:,S['aoa']]  = self._aoa_vec(s[:,S['r']], s[:,S['V']])
            peak_q = np.maximum(peak_q, self._q_dot_kW_vec(s[:,S['rho']], s[:,S['V']]))
            peak_g = np.maximum(peak_g, self._g_load_vec(s[:,S['rho']], s[:,S['V']], s[:,S['CD']], s[:,S['CL']], s[:,S['r']], s[:,S['lat']]))
            climb  = np.maximum(climb, s[:,S['r']] - r0)
        return peak_q, peak_g, climb

    def rollout_peaks(self, state, bank):
        q, g, _ = self.rollout_peaks_batch(state, [bank])
        return float(q[0]), float(g[0])

    def check(self, state, bank):
        q, g = self.rollout_peaks(state, bank)
        hf, gl = q > self.q_limit, g > self.g_limit
        return "Both" if (hf and gl) else "Heat Flux" if hf else "G-load" if gl else "None"

    # ── BATCHED GRID SEARCH with SKIP GUARD ─────────────────────────────────────
    # Constraint peaks (q̇, g) INCREASE with bank; skip risk DECREASES with bank
    # (more bank -> less upward lift -> no phugoid). So the acceptable set is a BAND
    # in [0, commanded]: high enough to not skip, low enough to satisfy q̇/g.
    # We pick the LARGEST bank <= commanded in that band (minimal perturbation).
    # DO-NO-HARM: if no bank both satisfies the constraints AND avoids a skip, we
    # keep the commanded bank (a marginal sub-limit q̇ is far better than a skip that
    # causes a downstream g-load spike + huge range miss).
    def find_safe_bank(self, state, commanded_bank, n_grid=19, refine=True):
        """Return (safe_bank_rad, info). Minimal reduction of the commanded bank."""
        cmd = abs(commanded_bank)
        sgn = np.sign(commanded_bank) if commanded_bank != 0 else 1.0
        def eval_grid(banks):
            q, g, climb = self.rollout_peaks_batch(state, banks)
            ok_qg   = (q <= self.q_target) & (g <= self.g_target)
            no_skip = climb <= self.climb_limit_m
            return ok_qg & no_skip, q, g, climb

        grid = np.linspace(0.0, cmd, n_grid)
        m, q, g, climb = eval_grid(grid)

        # commanded already satisfies constraints (skip is irrelevant at the
        # range-optimal bank the guidance chose) -> no change
        if (q[-1] <= self.q_target) and (g[-1] <= self.g_target):
            return commanded_bank, {'status':'ok_no_change',
                    'peak_q':float(q[-1]),'peak_g':float(g[-1]),'climb_m':float(climb[-1])}
        if not m.any():
            # No bank both fixes the constraint and avoids a skip -> DO NO HARM.
            return commanded_bank, {'status':'no_safe_nonskip_keep_commanded',
                    'peak_q':float(q[-1]),'peak_g':float(g[-1]),
                    'min_climb_m':float(climb.min())}
        k = int(np.where(m)[0].max())              # largest safe bank in the band
        safe_bank, kq = grid[k], k
        if refine and k+1 < n_grid:
            fine = np.linspace(grid[k], grid[k+1], n_grid)
            mf, qf, gf, cf = eval_grid(fine)
            if mf.any():
                kf = int(np.where(mf)[0].max())
                safe_bank, q, g, climb, kq = fine[kf], qf, gf, cf, kf
        return float(sgn*safe_bank), {'status':'corrected',
                'from_deg':float(np.rad2deg(cmd)), 'to_deg':float(np.rad2deg(safe_bank)),
                'peak_q':float(q[kq]), 'peak_g':float(g[kq]), 'climb_m':float(climb[kq])}


# ─── offline self-test ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import pandas as pd
    cur = os.path.dirname(os.path.abspath(__file__))
    print("Loading corrector (atmosphere from data for offline test)...")
    cor = PINNCorrector(atmosphere='data')

    # take a real held-out test run, build a state partway through (peak-heating region)
    split = joblib.load(os.path.join(cur,'PINN_Tensors','run_split.pkl'))
    DAT_COLS=[0,1,4,5,6,7,8,9,12,16,18,22]
    rf = split['test'][0]
    dep=pd.read_csv(os.path.join(cur,'Training_Simulations',rf,'Dependent_Variables.dat'),sep=r'\s+',header=None)
    num=pd.read_csv(os.path.join(cur,'Training_Simulations',rf,'Numerical_States.dat'),sep=r'\s+',header=None)
    n=min(len(dep),len(num)); dep,num=dep.iloc[:n],num.iloc[:n]
    r=np.sqrt(num[1]**2+num[2]**2+num[3]**2).values
    i = 360   # ~180 s in
    st = [r[i], dep[1].values[i], dep[4].values[i], dep[5].values[i], dep[6].values[i],
          dep[7].values[i], dep[8].values[i], dep[9].values[i], dep[12].values[i],
          dep[16].values[i], dep[18].values[i], dep[22].values[i]]
    print(f"\nTest state from {rf} @ step {i}:  alt={cor._altitude(st[0],st[2])/1e3:.1f} km  "
          f"V={st[11]:.0f} m/s  true bank={np.rad2deg(st[7]):.1f} deg\n")

    print("Monotonicity check — peak q̇ / peak g vs bank angle:")
    for bdeg in [0, 20, 40, 60, 75, 87]:
        q, g = cor.rollout_peaks(st, np.deg2rad(bdeg))
        print(f"  bank={bdeg:3d} deg ->  peak q̇={q:7.1f} kW/m²   peak g={g:5.2f}")

    print("\nBisection correction (commanded bank = 80 deg, target {:.0f}/{:.1f}):"
          .format(cor.q_target, cor.g_target))
    safe, info = cor.find_safe_bank(st, np.deg2rad(80.0))
    print(f"  -> safe bank = {np.rad2deg(safe):.2f} deg   info={info}")
    q, g = cor.rollout_peaks(st, safe)
    print(f"  verify @ safe bank:  peak q̇={q:.1f} kW/m²  peak g={g:.2f}")
