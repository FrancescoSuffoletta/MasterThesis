"""
PINN — SHORT-HORIZON rollout sweep
──────────────────────────────────────
A predictor-corrector re-plans periodically, so the surrogate only needs to be
accurate over one guidance update interval. This sweeps the look-ahead horizon
and measures how prediction error grows, to pick a decent Δt for acting on the
control angles before a constraint breach.

Method: launch MANY short rollouts from points spaced along held-out test
trajectories. True AoA/bank fed in; density/Mach from r->lookup; q_dot/g_load
computed analytically from the rolled-out state. Vectorised over all windows.

Reports, per horizon (10..200 s): median & 90th-pct error in altitude, velocity,
q_dot, g_load, plus peak-within-window q_dot/g_load error (what matters for
breach detection). Saves curve + CSV to .../rollout/horizon_sweep.*
"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, pandas as pd, torch, joblib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
from model import FlowMapMLP

HORUS_RN,C_STAR,VC,N_HF,M_HF,RHO_0 = 0.8,1.1097e8,7.9054e3,0.5,3.15,1.225
HORUS_MASS,HORUS_AREA = 26_029.0,110.0
MU,RE,J2 = 3.986004418e14,6_371.0e3,0.00108263
Q_LIMIT,G_LIMIT = 530.0,3.0
DT = 0.5

def q_dot_kW(rho,V):
    V = np.clip(V, 1.0, None)                      # avoid V**3.15 of negatives
    return (C_STAR/HORUS_RN**N_HF*(np.clip(rho,1e-30,None)/RHO_0)**(1-N_HF)*(V/VC)**M_HF)/1000.0
def g_load(rho,V,CD,CL,r,lat):
    qS=0.5*rho*V**2*HORUS_AREA; L=qS*CL; D=qS*CD
    g_r=(MU/r**2)*(1-1.5*J2*(RE/r)**2*(3*np.sin(lat)**2-1))
    g_lat=3*(MU/r**2)*(RE/r)**2*J2*np.sin(lat)*np.cos(lat)
    return np.sqrt(L**2+D**2)/(HORUS_MASS*np.sqrt(g_r**2+g_lat**2))

cur=os.path.dirname(os.path.abspath(__file__))
data_dir=os.path.join(cur,'Training_Simulations')
model_dir=os.path.join(cur,'PINN_Model','FlowMap_h256_d6')
scal_dir=os.path.join(cur,'PINN_Scalers')
out_dir=os.path.join(model_dir,'rollout'); os.makedirs(out_dir,exist_ok=True)

IN_FEATS=['delta_t','r','mach','latitude','longitude','heading','flight_path','aoa','bank','density','CD','CL','airspeed']
OUT_FEATS=['r','latitude','longitude','heading','flight_path','CD','CL','airspeed','q_dot','g_load']
DAT_COLS=[0,1,4,5,6,7,8,9,12,16,18,22]
DAT_NAMES=['time','mach','latitude','longitude','heading','flight_path','aoa','bank','density','CD','CL','airspeed']

device=torch.device('cpu')
model=FlowMapMLP(13,10,256,6,0.05).to(device)
model.load_state_dict(torch.load(os.path.join(model_dir,'model_state_dict_best.pt'),map_location=device,weights_only=True))
model.eval()
in_sc={f:joblib.load(os.path.join(scal_dir,f'{f}_scaler.pkl')) for f in IN_FEATS}
out_sc={f:joblib.load(os.path.join(scal_dir,f'{f}_delta_scaler.pkl')) for f in OUT_FEATS}
in_scale=np.array([in_sc[f].scale_[0] for f in IN_FEATS]); in_min=np.array([in_sc[f].min_[0] for f in IN_FEATS])
out_scale=np.array([out_sc[f].scale_[0] for f in OUT_FEATS]); out_min=np.array([out_sc[f].min_[0] for f in OUT_FEATS])

split=joblib.load(os.path.join(cur,'PINN_Tensors','run_split.pkl'))

def read_run(rf):
    rp=os.path.join(data_dir,rf)
    dep=pd.read_csv(os.path.join(rp,'Dependent_Variables.dat'),sep=r'\s+',header=None)
    num=pd.read_csv(os.path.join(rp,'Numerical_States.dat'),sep=r'\s+',header=None)
    n=min(len(dep),len(num)); dep,num=dep.iloc[:n],num.iloc[:n]
    r=np.sqrt(num[1]**2+num[2]**2+num[3]**2).values
    df=dep[DAT_COLS].copy(); df.columns=DAT_NAMES; df.insert(1,'r',r)
    return df

# atmosphere lookup
rr,dd,ss=[],[],[]
for rf in split['train'][:120]:
    df=read_run(rf); rr.append(df['r'].values); dd.append(df['density'].values); ss.append(df['airspeed'].values/df['mach'].values)
rr=np.concatenate(rr); dd=np.concatenate(dd); ss=np.concatenate(ss)
o=np.argsort(rr); rr,dd,ss=rr[o],dd[o],ss[o]; log_dd=np.log(np.clip(dd,1e-30,None))
atmo_rho=lambda r: np.exp(np.interp(r,rr,log_dd))
atmo_sos=lambda r: np.interp(r,rr,ss)

# ─── Build windows ────────────────────────────────────────────────────────────
HOR_S = [10,25,50,75,100,150,200]            # seconds
Hmax  = int(max(HOR_S)/DT)                    # steps
STRIDE= 50                                    # new window every 25 s
N_RUNS= 25
SCOLS=['r','latitude','longitude','heading','flight_path','CD','CL','airspeed']

init=[]; aoaW=[]; bankW=[]; rho0=[]; mach0=[]
true_alt=[]; true_V=[]; true_q=[]; true_g=[]
for rf in split['test'][:N_RUNS]:
    df=read_run(rf); N=len(df)
    if N < Hmax+2: continue
    tq=q_dot_kW(df['density'].values,df['airspeed'].values)
    tg=g_load(df['density'].values,df['airspeed'].values,df['CD'].values,df['CL'].values,df['r'].values,df['latitude'].values)
    for s in range(0,N-Hmax-1,STRIDE):
        init.append([df[c].iloc[s] for c in SCOLS])
        aoaW.append(df['aoa'].values[s:s+Hmax]); bankW.append(df['bank'].values[s:s+Hmax])
        rho0.append(df['density'].iloc[s]); mach0.append(df['mach'].iloc[s])
        true_alt.append((df['r'].values[s:s+Hmax+1]-RE)/1e3)
        true_V.append(df['airspeed'].values[s:s+Hmax+1])
        true_q.append(tq[s:s+Hmax+1]); true_g.append(tg[s:s+Hmax+1])

S=np.array(init,dtype=np.float64)             # [W,8]
aoaW=np.array(aoaW); bankW=np.array(bankW)    # [W,Hmax]
rho=np.array(rho0); mach=np.array(mach0)      # [W]
true_alt=np.array(true_alt); true_V=np.array(true_V); true_q=np.array(true_q); true_g=np.array(true_g)
W=len(S); print(f"Windows: {W}  (from {N_RUNS} test runs)  Hmax={Hmax} steps = {max(HOR_S)} s\n",flush=True)

# ─── Vectorised rollout of all windows to Hmax ────────────────────────────────
pred_alt=np.zeros((W,Hmax+1)); pred_V=np.zeros((W,Hmax+1))
pred_q=np.zeros((W,Hmax+1));  pred_g=np.zeros((W,Hmax+1))
def store(k):
    pred_alt[:,k]=(S[:,0]-RE)/1e3; pred_V[:,k]=S[:,7]
    pred_q[:,k]=q_dot_kW(rho,S[:,7])
    pred_g[:,k]=g_load(rho,S[:,7],S[:,5],S[:,6],S[:,0],S[:,1])
store(0)
with torch.no_grad():
    for k in range(Hmax):
        X=np.column_stack([np.full(W,DT),S[:,0],mach,S[:,1],S[:,2],S[:,3],S[:,4],
                           aoaW[:,k],bankW[:,k],rho,S[:,5],S[:,6],S[:,7]])
        Xn=(X*in_scale+in_min).astype(np.float32)
        yn=model(torch.from_numpy(Xn)).numpy()
        d=(yn-out_min)/out_scale
        S[:,0]+=d[:,0]; S[:,1]+=d[:,1]; S[:,2]+=d[:,2]; S[:,3]+=d[:,3]
        S[:,4]+=d[:,4]; S[:,5]+=d[:,5]; S[:,6]+=d[:,6]; S[:,7]+=d[:,7]
        rho=atmo_rho(S[:,0]); mach=S[:,7]/atmo_sos(S[:,0])
        store(k+1)

# ─── Errors vs horizon ────────────────────────────────────────────────────────
rows=[]
for hs in HOR_S:
    h=int(hs/DT)
    ae=np.abs(pred_alt[:,h]-true_alt[:,h])
    ve=np.abs(pred_V[:,h]-true_V[:,h])
    qe=np.abs(pred_q[:,h]-true_q[:,h])
    ge=np.abs(pred_g[:,h]-true_g[:,h])
    # peak-within-window error (breach detection relevance)
    qpk=np.abs(pred_q[:,:h+1].max(1)-true_q[:,:h+1].max(1))
    gpk=np.abs(pred_g[:,:h+1].max(1)-true_g[:,:h+1].max(1))
    # breach-flag agreement within window
    qfa=(( pred_q[:,:h+1].max(1)>Q_LIMIT)==(true_q[:,:h+1].max(1)>Q_LIMIT)).mean()*100
    gfa=(( pred_g[:,:h+1].max(1)>G_LIMIT)==(true_g[:,:h+1].max(1)>G_LIMIT)).mean()*100
    rows.append(dict(horizon_s=hs,
        alt_med=np.median(ae), alt_p90=np.percentile(ae,90),
        V_med=np.median(ve),   V_p90=np.percentile(ve,90),
        q_med=np.median(qe),   q_p90=np.percentile(qe,90), q_peak_med=np.median(qpk),
        g_med=np.median(ge),   g_p90=np.percentile(ge,90), g_peak_med=np.median(gpk),
        q_flag_agree=qfa, g_flag_agree=gfa))
res=pd.DataFrame(rows); res.to_csv(os.path.join(out_dir,'horizon_sweep.csv'),index=False)
pd.set_option('display.width',220,'display.max_columns',30)
print("Error vs look-ahead horizon (median unless noted):\n")
print(res.to_string(index=False,float_format=lambda v:f'{v:.2f}'))

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(2,2,figsize=(12,7))
H=res.horizon_s
ax[0,0].plot(H,res.alt_med,'o-',label='median'); ax[0,0].plot(H,res.alt_p90,'s--',label='90th pct')
ax[0,0].set(title='Altitude error [km]',xlabel='horizon [s]'); ax[0,0].legend(); ax[0,0].grid(alpha=.3)
ax[0,1].plot(H,res.V_med,'o-'); ax[0,1].plot(H,res.V_p90,'s--')
ax[0,1].set(title='Velocity error [m/s]',xlabel='horizon [s]'); ax[0,1].grid(alpha=.3)
ax[1,0].plot(H,res.q_med,'o-',label='inst median'); ax[1,0].plot(H,res.q_peak_med,'^--',label='peak-in-window')
ax[1,0].set(title='Heat flux error [kW/m²]  (limit 530)',xlabel='horizon [s]'); ax[1,0].legend(); ax[1,0].grid(alpha=.3)
ax[1,1].plot(H,res.g_med,'o-',label='inst median'); ax[1,1].plot(H,res.g_peak_med,'^--',label='peak-in-window')
ax[1,1].set(title='g-load error [g]  (limit 3)',xlabel='horizon [s]'); ax[1,1].legend(); ax[1,1].grid(alpha=.3)
fig.suptitle(f'Short-horizon rollout error growth  ({W} windows, {N_RUNS} held-out runs)')
fig.tight_layout(); fig.savefig(os.path.join(out_dir,'horizon_sweep.png'),dpi=130); plt.close(fig)
print(f"\nSaved horizon_sweep.png + .csv to {out_dir}")
