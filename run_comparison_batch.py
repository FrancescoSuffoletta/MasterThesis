"""
AI-vs-baseline comparison runner  (PINN corrector ON vs OFF)
══════════════════════════════════════════════════════════════
The dedicated equivalent of Simulation_Runs.py for the constraint-check study.

Runs N PAIRED HORUS-2B re-entry simulations per wind setting:
  • "PINN_baseline"  -> pure predictor-corrector (NO AI interface)
  • "PINN"           -> same guidance + AI (PINN) constraint corrector

Each pair uses the SAME random seed, so both modes get the IDENTICAL dispersed
initial state (Monte-Carlo, perturb_initial_conditions) and wind realisation
-> a clean PAIRED A/B test that isolates the effect of the AI corrector.

MEMORY-SAFE: each simulation runs in its OWN SUBPROCESS, so the OS reclaims all
memory when it exits. This prevents the accumulation that otherwise crashes long
TUDAT batches, and makes the batch crash-proof (a failed run is logged and skipped).

──────────────────────────── WHERE RESULTS GO ────────────────────────────
ComparisonResults/
├── comparison_metrics.csv     one row per run (the comparison data, incl. compute time)
├── comparison_summary.txt     human-readable AI-vs-baseline table
├── comparison_plots.png       violation rates + peak q̇/g histograms
└── failures.log               any runs that errored / timed out (batch keeps going)
SimulationOutput/{Wind|NoWind}/PINN{,_baseline}/{Success,Acceptable,Fail}/run_*
    full per-run .dat trajectories (written by produce_simulation as usual)

────────────────────────────── HOW TO RUN ────────────────────────────────
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONUTF8=1 python run_comparison_batch.py --nowind
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONUTF8=1 python run_comparison_batch.py --wind
  python run_comparison_batch.py --both        # no-wind batch, then wind batch
  python run_comparison_batch.py --analyze     # just (re)build summary+plots from CSV
  optional:  --n 250   (paired runs; default 250)   --timeout 900   (per-sim seconds)

RESUMABLE: pairs already in the CSV are skipped, so stop/restart freely.
"""
import os, sys, csv, time, argparse, multiprocessing as mp

# NOTE: TUDAT (Simulation_Runs) is intentionally NOT imported at module top, so the
# PARENT process stays light and stable. Only the worker subprocess imports it.
import numpy as np

CUR     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CUR, 'ComparisonResults')
CSV     = os.path.join(OUT_DIR, 'comparison_metrics.csv')
BASE_SEED = 20260626

MODES = [('PINN_baseline', 'baseline (no AI)'),
         ('PINN',          'AI corrector')]


# ─── Worker: runs exactly ONE simulation in a fresh subprocess ──────────────────
def _run_one(run_id, gtype, wind, csv_path, seed):
    import numpy as _np
    _np.random.seed(seed)                       # identical MC dispersion within a pair
    from Simulation_Runs import produce_simulation
    produce_simulation(run_id, gtype, wind, summary_csv=csv_path)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    for w in ('Wind', 'NoWind'):
        for g in ('PINN', 'PINN_baseline'):
            for c in ('Success', 'Acceptable', 'Fail'):
                os.makedirs(os.path.join(CUR, 'SimulationOutput', w, g, c), exist_ok=True)


def done_keys():
    """(run, guidance_type) already recorded -> resumability (run id encodes wind)."""
    have = set()
    if os.path.exists(CSV):
        with open(CSV, newline='') as f:
            for row in csv.DictReader(f):
                have.add((row['run'], row['guidance_type']))
    return have


def _log_failure(run_id, gtype, wind, msg):
    """Never let logging crash the batch (this was the previous failure mode)."""
    try:
        with open(os.path.join(OUT_DIR, 'failures.log'), 'a') as fl:
            fl.write(f"{run_id}\t{gtype}\twind={wind}\t{msg}\n")
    except Exception:
        pass
    print(f"  !! {run_id} [{gtype}] FAILED: {msg}", flush=True)


def run_batch(wind, n_runs, timeout):
    ensure_dirs()
    ctx = mp.get_context('spawn')               # fresh interpreter per run (Windows default)
    tag = 'w' if wind else 'nw'
    have = done_keys()
    t0 = time.time()
    print(f"\n=== Comparison batch | wind={wind} | {n_runs} paired runs | "
          f"per-sim subprocess (timeout {timeout}s) ===", flush=True)
    for i in range(n_runs):
        run_id = f"cmp_{tag}_{i:04d}"
        for gtype, label in MODES:
            if (run_id, gtype) in have:
                continue
            p = ctx.Process(target=_run_one,
                            args=(run_id, gtype, wind, CSV, BASE_SEED + i))
            p.start()
            p.join(timeout)
            if p.is_alive():                    # hung -> kill and move on
                p.terminate(); p.join()
                _log_failure(run_id, gtype, wind, f"TIMEOUT after {timeout}s")
            elif p.exitcode != 0:               # crashed (MemoryError/segfault/etc.)
                _log_failure(run_id, gtype, wind, f"exit code {p.exitcode}")
        done = i + 1
        el = time.time() - t0
        print(f"[{tag} pair {done:3d}/{n_runs}] {run_id} | "
              f"{el/done:.1f}s/pair | ETA {(n_runs-done)*el/done/60:.0f} min", flush=True)
    print(f"Batch (wind={wind}) done in {(time.time()-t0)/60:.1f} min", flush=True)


# ─── Analysis (parent-side, no TUDAT needed) ────────────────────────────────────
def _load():
    if not os.path.exists(CSV):
        return []
    return list(csv.DictReader(open(CSV, newline='')))


def _table(rows, wind_filter=None):
    if wind_filter is not None:
        rows = [r for r in rows if int(r['wind']) == int(wind_filter)]
    b = [r for r in rows if r['guidance_type'] == 'PINN_baseline']
    a = [r for r in rows if r['guidance_type'] == 'PINN']
    def frac(rs, k): return 100.0*sum(int(r[k]) for r in rs)/len(rs) if rs else float('nan')
    def favg(rs, k): return float(np.mean([float(r[k]) for r in rs])) if rs else float('nan')
    def fmax(rs, k): return float(np.max([float(r[k]) for r in rs])) if rs else float('nan')
    def fmed(rs, k): return float(np.median([float(r[k]) for r in rs])) if rs else float('nan')
    def cls(rs, c):  return 100.0*sum(r['classification']==c for r in rs)/len(rs) if rs else float('nan')
    have_time = bool(rows) and ('propagation_time_s' in rows[0])
    L = []
    title = 'ALL' if wind_filter is None else ('WIND' if wind_filter else 'NO-WIND')
    L.append("="*72)
    L.append(f"  {title:62s}")
    L.append("="*72)
    L.append(f"{'metric':40s}{'baseline':>15s}{'AI':>15s}")
    L.append("-"*72)
    L.append(f"{'n runs':40s}{len(b):>15d}{len(a):>15d}")
    rowspec = [
        ('heat-flux violation rate [%]', lambda rs: frac(rs,'heat_flux_violation')),
        ('g-load violation rate [%]',    lambda rs: frac(rs,'g_load_violation')),
        ('reached-target rate [%]',      lambda rs: frac(rs,'reached_target')),
        ('mean peak heat flux [kW/m^2]', lambda rs: favg(rs,'max_heat_flux_kW')),
        ('max  peak heat flux [kW/m^2]', lambda rs: fmax(rs,'max_heat_flux_kW')),
        ('mean peak g-load [-]',         lambda rs: favg(rs,'max_g_load')),
        ('max  peak g-load [-]',         lambda rs: fmax(rs,'max_g_load')),
        ('mean final angular dist [deg]',lambda rs: favg(rs,'final_angular_distance_deg')),
        ('Success [%]',    lambda rs: cls(rs,'Success')),
        ('Acceptable [%]', lambda rs: cls(rs,'Acceptable')),
        ('Fail [%]',       lambda rs: cls(rs,'Fail')),
    ]
    if have_time :
        rowspec += [
            ('mean compute time / sim [s]',   lambda rs: favg(rs,'propagation_time_s')),
            ('median compute time / sim [s]', lambda rs: fmed(rs,'propagation_time_s')),
        ]
    for name, fn in rowspec:
        L.append(f"{name:40s}{fn(b):>15.2f}{fn(a):>15.2f}")
    bmap={r['run']:r for r in b}; amap={r['run']:r for r in a}
    paired=[r for r in bmap if r in amap]
    if paired:
        viol=lambda r:int(r['heat_flux_violation']) or int(r['g_load_violation'])
        bvio=[r for r in paired if viol(bmap[r])]
        fixed=[r for r in bvio if not viol(amap[r])]
        newv=[r for r in paired if not viol(bmap[r]) and viol(amap[r])]
        L.append("-"*72)
        L.append(f"paired pairs: {len(paired)} | baseline violated: {len(bvio)} | "
                 f"AI fixed: {len(fixed)} | AI new-violation: {len(newv)}")
    L.append("="*72)
    return "\n".join(L), b, a


def analyze():
    rows = _load()
    if not rows:
        print("No CSV data yet."); return
    winds = sorted({int(r['wind']) for r in rows})
    blocks = []
    if len(winds) > 1:
        blocks.append(_table(rows, None)[0])
    for w in winds:
        blocks.append(_table(rows, w)[0])
    text = "\n\n".join(blocks)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, 'comparison_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    _plots(rows, winds)
    print(f"\nSaved -> {os.path.join(OUT_DIR,'comparison_summary.txt')}")
    print(f"Saved -> {os.path.join(OUT_DIR,'comparison_plots.png')}")


def _plots(rows, winds):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    w = winds[0] if len(winds) == 1 else None
    _, b, a = _table(rows, w)
    if not b or not a: return
    bq=[float(r['max_heat_flux_kW']) for r in b]; aq=[float(r['max_heat_flux_kW']) for r in a]
    bg=[float(r['max_g_load']) for r in b];       ag=[float(r['max_g_load']) for r in a]
    def rate(rs,k): return 100.0*sum(int(r[k]) for r in rs)/len(rs) if rs else 0
    fig, ax = plt.subplots(1,3, figsize=(15,4.3))
    cats=['HF viol','g viol','Fail','Reached']
    bvals=[rate(b,'heat_flux_violation'),rate(b,'g_load_violation'),
           100*sum(r['classification']=='Fail' for r in b)/len(b),rate(b,'reached_target')]
    avals=[rate(a,'heat_flux_violation'),rate(a,'g_load_violation'),
           100*sum(r['classification']=='Fail' for r in a)/len(a),rate(a,'reached_target')]
    x=np.arange(len(cats)); ww=0.38
    ax[0].bar(x-ww/2,bvals,ww,label='baseline'); ax[0].bar(x+ww/2,avals,ww,label='AI')
    ax[0].set_xticks(x); ax[0].set_xticklabels(cats); ax[0].set_ylabel('%')
    ax[0].set_title('Outcome rates'); ax[0].legend(); ax[0].grid(alpha=.3,axis='y')
    ax[1].hist(bq,bins=30,alpha=.6,label='baseline'); ax[1].hist(aq,bins=30,alpha=.6,label='AI')
    ax[1].axvline(530,color='r',ls='--',label='530 limit')
    ax[1].set_title('Peak heat flux [kW/m²]'); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].hist(bg,bins=30,alpha=.6,label='baseline'); ax[2].hist(ag,bins=30,alpha=.6,label='AI')
    ax[2].axvline(3.0,color='r',ls='--',label='3 g limit')
    ax[2].set_title('Peak g-load [-]'); ax[2].legend(); ax[2].grid(alpha=.3)
    ttl = 'all' if w is None else ('wind' if w else 'no-wind')
    fig.suptitle(f'AI corrector vs baseline ({ttl}, n={len(b)} pairs)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,'comparison_plots.png'),dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    mp.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument('--wind',    action='store_true', help='run the WIND batch')
    p.add_argument('--nowind',  action='store_true', help='run the NO-WIND batch')
    p.add_argument('--both',    action='store_true', help='run no-wind then wind')
    p.add_argument('--analyze', action='store_true', help='only rebuild summary+plots from CSV')
    p.add_argument('--n', type=int, default=250, help='number of paired runs (default 250)')
    p.add_argument('--timeout', type=int, default=900, help='per-sim timeout in seconds (default 900)')
    p.add_argument('--csv', type=str, default=None,
                   help='CSV filename inside ComparisonResults (default comparison_metrics.csv). '
                        'Use a new name to re-run without touching previous results.')
    args = p.parse_args()

    if args.csv:                       # redirect all reads/writes to the chosen CSV
        CSV = os.path.join(OUT_DIR, args.csv)

    if args.analyze:
        ensure_dirs(); analyze()
    else:
        if args.both:
            run_batch(False, args.n, args.timeout); run_batch(True, args.n, args.timeout)
        elif args.wind:
            run_batch(True, args.n, args.timeout)
        else:
            run_batch(False, args.n, args.timeout)
        analyze()
