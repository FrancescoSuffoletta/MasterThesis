# HORUS-2B Re-entry Guidance with a PINN Constraint Corrector

This repository contains the codes for the development of the Master Thesis by the
author, **Francesco Benigno Suffoletta**. The intellectual property of what is here
contained pertains to the aforementioned author. The codes have been developed also
using external libraries, such as TUDAT (TU Delft Astrodynamics Toolbox), PyTorch,
poliastro, scikit-learn, etc.

The project couples a classical **numerical predictor-corrector entry guidance** for
the winged re-entry vehicle **HORUS-2B** with a **physics-informed neural network
(PINN) flow-map surrogate** used as a *fast onboard integrator*: at guidance rate it
propagates the current state 50 s into the future, predicts whether the path
constraints (stagnation-point heat flux, aerodynamic load factor) are about to be
violated, and — if so — minimally corrects the commanded bank angle before the
violation materialises.

---

## 1. Repository structure

| File / folder | Role |
|---|---|
| `Utils.py` | Simulation utilities: environment, vehicle, aero database, integrator, output writing |
| `MainGuidance.py` | `HORUSGuidanceAndTrim` — the complete guidance & trim law (all modes) |
| `HelperGuidance.py` | Reduced copy of the guidance used *inside* the onboard predictor propagations |
| `pinn_corrector.py` | `PINNCorrector` — the PINN 50 s look-ahead + bank-correction logic (TUDAT-independent) |
| `Simulation_Runs.py` | `produce_simulation()` — sets up and runs one full TUDAT re-entry simulation |
| `run_comparison_batch.py` | Batch driver for the paired AI-vs-baseline Monte-Carlo study |
| `PINN_Data_Preprocessing.py` | Builds the PINN training set from recorded simulations |
| `PINN_train.py` | Trains the PINN flow-map surrogate (physics-informed loss framework) |
| `model.py` | `FlowMapMLP` network + `TorchMinMaxScaler` (differentiable scaler) |
| `PINN_rollout_validation.py` | Full-trajectory autoregressive rollout validation of the surrogate |
| `PINN_rollout_horizon.py` | Error-vs-look-ahead-horizon sweep (justifies the 50 s horizon) |
| `HelperPlot.py`, `PostProcessing.py` | Result visualisation / post-processing |
| `Testing.py` | Single-simulation debug entry point |
| `Training_Simulations/` | 3202 recorded TUDAT re-entry trajectories (PINN training data) |
| `PINN_Tensors/`, `PINN_Scalers/` | Preprocessed train/val/test tensors and fitted scalers |
| `PINN_Model/FlowMap_h256_d6/` | **The selected trained model** (`model_state_dict_best.pt`) + training artefacts |
| `SimulationOutput/` | Per-run trajectory outputs, sorted by wind / guidance / outcome |
| `ComparisonResults/` | Metrics CSVs, summary tables and plots of the AI-vs-baseline study |
| `body_coefficients/`, `ctrl_surf_coefficients/` | HORUS-2B aerodynamic database (CD0/CL0/Cm0, body-flap & elevon ΔCm) |
| `Unit Verification Tests/`, `Verification files/` | Software verification tests against reference data |
| `Archive_Superseded/` | Earlier model/pipeline iterations kept for reference (not used) |

---

## 2. Simulation environment (`Utils.py`)

Each simulation propagates HORUS-2B (26 029 kg, 110 m² reference area, 0.8 m nose
radius) as a 3-DOF point mass with attitude-dependent aerodynamics:

- **Earth**: oblate spheroid (flattening 1/298.257), simple rotation model,
  spherical-harmonic gravity, **US76 atmosphere**.
- **Wind** (optional): `wind_function()` adds a sinusoidal wind field in the vertical
  frame — amplitudes 5 m/s (meridional), 5 m/s (zonal), 1 m/s (vertical), ~60 s
  period. The vertical component is kept small because real vertical winds are ≈ 0
  and it maps almost directly onto the flight-path angle.
- **Aerodynamics**: tabulated HORUS-2B coefficients interpolated in (AoA, Mach),
  plus control-surface increments interpolated in (AoA, deflection, Mach).
- **Integrator**: fixed-step RK4, **dt = 0.5 s** (`get_integrator_settings`).
- **Termination**: altitude < 25 km, or angular distance to target < 0.75°, or
  1-day timeout.
- **Initial state** (nominal): h = 122 km, lat −22.3°, lon −106.7°, V = 7435.5 m/s,
  γ = −1.43°, χ = 70.75°. Monte-Carlo batches perturb it via
  `perturb_initial_conditions()` (Gaussian: σ = 50 m, 0.033°, 5 m/s, 0.033°, 0.033°).
- **Path constraints** (mission requirements):
  **q̇ ≤ 530 kW/m²** (stagnation heat flux, Chapman-type relation) and
  **n ≤ 3 g** (aerodynamic load factor).

---

## 3. Guidance (`MainGuidance.py` — `HORUSGuidanceAndTrim`)

Called by TUDAT at every integrator step through `getAerodynamicAngles()`, the class
produces the angle of attack, the bank angle and the pitch trim.

### 3.1 Angle of attack — energy law
`Update_AoA()` interpolates linearly with the vehicle's specific energy
`E = 1/r − V²/2` from the **maximum AoA (40°)** at high energy down to the
**near-maximum-efficiency AoA (11.5°)** for the approach, with the rate limited to
**5°/s**.

### 3.2 Longitudinal guidance — numerical predictor-corrector
`Predictor_Corrector_cycle()` runs at a low rate (every 100 s early, every 20 s in
the terminal phase) and commands the **bank-angle magnitude**:

1. **Predictor** — `predictor_HORUS()` propagates a *copy* of the vehicle
   (`create_copy_bodies`, `HelperGuidance`) from the current state to the terminal
   conditions with the candidate constant bank, returning the integrated range,
   final energy, peak flight-path angle, peak g-load, peak heat flux and final Mach.
2. **Corrector** — a **secant iteration** (max 8 iterations, 0.1 % convergence
   threshold) drives the *range error* (range-to-go vs integrated range) to zero;
   in the terminal energy phase the iteration switches to the *final-energy error*.
3. **Safeguards** — a heat-flux pull-up (bank → 0 once, if the predicted peak
   exceeds 530 kW/m²), a **quasi-equilibrium-glide** bank floor (`get_sigma_QECG`),
   re-prediction loops that back the bank off while predicted peak g or final Mach
   exceed 3, an 85° hard cap, and NaN/Inf fallbacks.

### 3.3 Lateral guidance — bank-reversal deadband
The bank *sign* is handled by `evaluate_reversal()`: the heading error to the target
is compared against an angular-distance-dependent deadband (interpolated from the
`deadband_distance` / `heading_error` tables); crossing it triggers a **bank
reversal**, executed smoothly by `perform_bank_reversal()`.

### 3.4 Rate limiting
Every commanded bank change is shaped by `perform_maneuver()` as a raised-cosine
transition whose period `compute_period_transition()` sizes so the bank rate never
exceeds **15°/s** (`max_sigma_dot`).

### 3.5 Pitch trim
At every step the aerodynamic pitching moment is cancelled (`Trim_Horus`): the
required ΔCm is inverted through the **body-flap** database (deflection −20°…+30°);
if insufficient, the **elevons** (±40°) provide the remainder.

### 3.6 Guidance types
| `guidance_type` | Meaning |
|---|---|
| `'Predictor-Corrector'` | Numerical predictor-corrector guidance only |
| `'PINN'` | Predictor-corrector **+ PINN constraint corrector** (the selected configuration) |
| `'PINN_baseline'` | Identical code path with the corrector **disabled** (paired A/B baseline) |
| `'Verification'` | Prescribed control-node profile for verification runs |

(`PINN`/`PINN_baseline` are kept as identifiers because all recorded results and
output folders use them; they denote the final selected PINN configuration.)
The constructor also accepts `use_corrector=False` to disable the corrector
programmatically.

---

## 4. The PINN flow-map surrogate

### 4.1 Task
Replace one step of numerical integration with a single network evaluation:

```
input  (13):  dt, r, Mach, lat, lon, heading χ, flight path γ, AoA, bank,
              density ρ, CD, CL, airspeed V          (state at time t)
output (10):  Δr, Δlat, Δlon, Δχ, Δγ, ΔCD, ΔCL, ΔV, Δq̇, Δn   (increments over dt)
```

Predicting *increments* (a flow map) rather than absolute states keeps the learning
problem well-conditioned: over one 0.5 s step the deltas span a tiny, nearly linear
range compared with the absolute states.

### 4.2 Data and preprocessing (`PINN_Data_Preprocessing.py`)
- **3202 recorded trajectories** (≈ 7.9 M state transitions) generated by the TUDAT
  RK4 propagator under dispersed initial conditions.
- **Split BY TRAJECTORY** — 70 % / 15 % / 15 % of whole runs (2241/480/481) go to
  train/val/test. This is essential: a per-timestep random split would place
  adjacent, nearly identical samples of the same trajectory in both train and test,
  making the test error measure interpolation instead of generalisation.
- **Scalers fit on the train split only** (MinMax to [−1, 1], one per feature); the
  run-to-split assignment is persisted (`run_split.pkl`) so trajectory-level
  validation can use genuinely unseen runs.
- q̇ and n are computed analytically at every timestep and their deltas added as
  supervised outputs.

### 4.3 Architecture (`model.py`)
`FlowMapMLP`: input projection (13 → 256) followed by **6 pre-activation residual
blocks** (Linear–LayerNorm–Tanh–Dropout–Linear–LayerNorm with skip connection) and a
linear head (256 → 10) — **802 314 parameters**. The residual structure lets the
network learn small corrections on top of a near-linear baseline, matching the
smooth dynamics. `TorchMinMaxScaler` re-implements the fitted sklearn scalers as
differentiable PyTorch buffers so the physics loss can operate in physical units
inside the computational graph.

### 4.4 Learning and the loss (`PINN_train.py`)
The optimised objective is the **data MSE**:

```
L = L_data
```

- **L_data**: MSE between predicted and recorded (normalised) deltas.

The training data are generated by a high-order physics propagator (RK45/RK4 with
aerodynamic coefficients re-evaluated at every substep), so the physics is embedded
*exactly* in the data. A one-step Euler residual with frozen coefficients is only an
approximation of that integrator — diagnostics measured its disagreement with the
recorded deltas at ≈ 400× the converged data loss, making an explicit physics-residual
term counterproductive. A **physics-residual diagnostic** (8 EoM terms: Δr, Δlat,
Δlon, ΔV, Δγ, Δχ, Δq̇, Δn) is still evaluated every epoch to monitor physical
consistency without entering the gradient. The physics enters the model through
(1) the physics-generated data, (2) the physics-structured input/output spaces, and
(3) the analytic constraint evaluation at inference (§5).

**Optimisation**: Adam (lr 5·10⁻³), batch 8192, `ReduceLROnPlateau`, gradient-norm
clipping (1.0), best-validation checkpointing, early stopping (patience 30).
The selected model early-stopped at epoch 160 with the best epoch at 130.

**Accuracy** (normalised MSE on *unseen trajectories*): test MSE **6.0·10⁻⁶**; the
constraint outputs are among the best predicted (q̇ 2.5·10⁻⁶, n 2.1·10⁻⁷).

### 4.5 Validation of the surrogate as an integrator
- **Full-trajectory autoregressive rollout** (`PINN_rollout_validation.py`): the
  model feeds back its own predictions for ~2400 steps. Errors compound (median
  altitude RMSE ≈ 7 km over a full entry) — an honest demonstration that a one-step
  surrogate cannot replace the full-trajectory propagator.
- **Horizon sweep** (`PINN_rollout_horizon.py`, ~1000 windows on held-out runs):
  error grows smoothly with look-ahead. At **50 s**: median altitude error ≈ 60 m,
  velocity ≈ 19 m/s, peak-q̇ error ≈ 1 kW/m², g-load breach-flag agreement 100 %.
  Beyond ~100–150 s the peak errors grow sharply. **This measurement fixes the
  corrector's design point: a 50 s look-ahead.**

---

## 5. The PINN constraint corrector (`pinn_corrector.py`)

`PINNCorrector` is the deployable product of the PINN — a fast, self-contained
50 s look-ahead breach predictor + bank corrector, called by the guidance every
**10 s** (outside bank reversals, within the operative energy window).

1. **Rollout**: from the current state it autoregressively applies the flow map for
   100 × 0.5 s steps with a *candidate* bank held constant, the AoA following its
   energy law, and density/Mach recomputed each step from a pre-tabulated US76
   (altitude → ρ, speed-of-sound) table.
2. **Constraints from physics, not from the network**: peak q̇ and peak n over the
   window are evaluated **analytically** from the rolled-out state (the network's
   Δq̇/Δn outputs are used only in training) — guaranteeing constraint/state
   consistency.
3. **Detection**: if the commanded bank keeps peaks under the **margined targets**
   (510 kW/m², 2.9 g — margins absorb the ~1 kW/m² surrogate error at 50 s), nothing
   happens.
4. **Correction — batched-grid bisection**: because the peaks increase monotonically
   with |bank| near the constraint peaks, the corrector evaluates a whole grid of
   candidate banks in **one batched rollout** (CPU batching is nearly free), takes
   the largest safe bank ≤ commanded (minimal perturbation of the range solution),
   and refines once — sub-degree precision in ~2 rollouts (~0.8 s). This replaced an
   earlier Newton iteration that required finite-difference derivatives (3 rollouts
   per step) and was destabilised by surrogate noise.
5. **Skip guard**: any candidate whose rollout *climbs* more than 500 m above the
   current altitude is rejected — an over-reduced bank converts lift into a phugoid
   skip that returns later as a worse g-load peak.
6. **Do-no-harm**: if no bank both satisfies the constraints *and* avoids a skip,
   the commanded bank is kept — a marginal, sub-limit heat-flux is preferred to a
   destabilising intervention.
7. Every accepted correction re-enters the guidance's rate-limited maneuver shaping
   (§3.4), so the 15°/s bank-rate constraint is always respected, and the AoA law is
   untouched.

---

## 6. The comparison study (`run_comparison_batch.py`)

Paired Monte-Carlo A/B test of the corrector: for each pair index *i*, the same
seed (`BASE_SEED + i`) generates identical dispersed initial conditions, and the
simulation is run once as `PINN_baseline` (no AI) and once as `PINN` (with AI) —
differences are attributable to the corrector alone. Each simulation runs in its own
subprocess (memory isolation; a crash/hang is logged and skipped) and the batch is
resumable.

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONUTF8="1"
python run_comparison_batch.py --nowind --n 250
python run_comparison_batch.py --wind   --n 250
python run_comparison_batch.py --analyze
```

### Results (250 pairs per regime, final configuration)

| metric | no-wind baseline | no-wind **AI** | wind baseline | wind **AI** |
|---|---|---|---|---|
| heat-flux violation rate | 9.2 % | **4.0 %** | 11.2 % | **5.2 %** |
| g-load violation rate | 0.4 % | 0.4 % | 0.4 % | 0.4 % |
| Fail rate | 9.2 % | **4.0 %** | 11.2 % | **5.2 %** |
| Success rate | 80.0 % | **80.4 %** | 82.8 % | **83.2 %** |
| mean peak q̇ [kW/m²] | 515.3 | 513.5 | 515.7 | 513.9 |
| **new violations introduced by AI** | — | **0** | — | **0** |
| median compute time / sim | 63 s | 102 s | 65 s | 108 s |

The corrector **halves the constraint violations and failure rate in both regimes,
introduces zero new violations across 500 paired tests, and slightly raises the
Success rate**, at ~1.7× computational cost. The residual worst-case peaks are
dominated by a single pathological dispersion (`*_0111`) whose baseline violates
both constraints as well — a finite-horizon energy-shift case discussed as a
limitation.

---

## 7. How to run the pipeline

```powershell
# 0) environment (Windows / conda 'tudat-space'); always set:
$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONUTF8="1"

# 1) single simulation (debug)  — edit guidance_type/wind in Simulation_Runs.main()
python Simulation_Runs.py

# 2) paired AI-vs-baseline batches  
python run_comparison_batch.py --nowind --n 250
```
python run_comparison_batch.py --analyze 

(If more simulations are desired to be compared, substitute the 250 number with the chosen one from above)

---

## 8. Known limitations & future work

- **Finite horizon**: the corrector optimises the next 50 s; on rare
  near-infeasible entries a local correction can shift a constraint peak beyond the
  horizon (seen once per 250 runs, on a dispersion whose baseline also fails). A
  coarse longer-horizon "do-no-harm" re-check is the natural extension.
- **Wind-blindness**: the surrogate is trained on no-wind dynamics and its rollout
  atmosphere is static US76; under strong winds its forecasts degrade (quantified in
  the study — with realistic wind levels the corrector remains strictly beneficial).
  A wind-robust variant (uncertainty-widened margins) or a wind-aware model are
  future options.
- **One-step training**: the model is trained teacher-forced on single steps;
  rollout-aware (multi-step) training would reduce the free-running drift and could
  extend the usable horizon.

---

*Superseded iterations of every component (earlier architectures, per-timestep
splits, the Newton-based corrector, the PCNN commanded-bank experiments and the old
model checkpoints) are preserved unmodified in `Archive_Superseded/` for
traceability.*
