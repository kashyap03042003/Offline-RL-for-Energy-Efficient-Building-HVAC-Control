# Offline Reinforcement Learning for Energy-Efficient Building Control — Technical Report

## 1. Problem

A commercial building's BMS has years of logs produced by a rule-based schedule plus operator overrides.
The question is whether a better HVAC controller can be learned **from those logs alone** (no online
exploration in an occupied building), and whether it stays safe when the weather, sensors, or equipment differ
from the logs.

**Objective** per 15-minute step: minimize electricity cost (TOU energy + daily-peak demand charge) subject to
occupant comfort (20–24 °C during weekday occupancy 07–19 h; 15–30 °C otherwise).
Reward = −(cost + 2·°C occupied violation + 0.5·°C unoccupied violation).

## 2. Setup

| Component | Choice |
|---|---|
| Building | 2R2C RC thermal model (air + envelope), 1-min substeps, 15-min control, hidden envelope state |
| HVAC | heat pump, COP(T_out), 40 kW capacity, 1.5 kW fan, P-controlled setpoint tracking |
| Actions | 10 discrete: OFF or setpoint ∈ {17, 19, 20, 21, 22, 23, 24, 25, 27} °C |
| Observation | 13 features (T_in, ΔT_in, T_out, solar, time, occupancy, weekday, time-to-occupancy, price now / next 2 h, 3 h T_out forecast, today's peak kW) |
| Dataset | 2,000 logged days = 192k transitions from a heterogeneous behavior policy (varied schedules, deadband days, overrides, 15% random commissioning days) |
| Baseline | BMS schedule: 22 °C occupied, 1 h pre-start, 16/29 °C setback |
| Evaluation | 500 Monte Carlo test days per scenario, common random numbers across policies, 3 training seeds, paired bootstrap 95% CI over days |
| Seeds | data 0 · validation 50,000 (all tuning) · test 900,000+ (reported) · OPE truth 123,456 |

Final checkpoints are reported. No test-set checkpoint picking.

## 3. Methods

* **BC / Filtered BC** — imitation of all data / of the top 25% days by return, stratified by weekday × outdoor-temperature quintile.
* **CQL(H)** — Double-DQN with a 2-critic ensemble (min target), n = 4 step returns truncated at day end, α = 0.1.
* **IQL** — expectile (τ = 0.7) value function + advantage-weighted policy extraction (β = 30).
* **Lagrangian CQL** — separate reward and comfort-cost critics. Conservatism pushes out-of-distribution cost *up*, and dual ascent on λ enforces a cost limit.
* **Model-based MPC** — 5-member bootstrap MLP ensemble predicting (ΔT_in, energy) with validation RMSE ≈ 0.14 °C. Exhaustive search over two 1-hour segments using the known schedule and tariff, with a 0.3 °C comfort margin. An uncertainty penalty and a terminal IQL value were tested on validation and did not help.
* **OOD fallback** — dynamics-ensemble disagreement on (s, π(s)). If it exceeds the 99th percentile of logged data, the controller hands over to the rule-based schedule.
* **Off-policy evaluation** — FQE, per-decision weighted importance sampling (BC as behavior model, ε = 0.1 smoothing), and weighted doubly-robust.

Hyperparameters were chosen by validation sweeps (`scripts/sweep.sh`, `runs_sweep*`).

## 4. Results

### 4.1 In-distribution

| Policy | kWh/day | Energy saved (95% CI) | Cost saved (95% CI) | Peak cut | Comfort |
|---|---|---|---|---|---|
| Rule-based | 45.3 | – | – | – | 100.0% |
| **MPC** | 41.8 | **7.6% (6.1–9.0)** | **17.0% (15.5–18.4)** | **25.7%** | 99.7% |
| IQL | 42.0 | 7.2% (6.4–8.0) | 6.9% (6.2–7.6) | 10.9% | 99.6% |
| IQL + fallback | 42.3 | 6.5% (5.9–7.1) | 5.9% (5.3–6.5) | 8.5% | 99.7% |
| CQL | 43.3 | 4.3% (3.5–5.1) | 5.0% (4.3–5.8) | 9.7% | 99.0% |
| Lagrangian CQL | 45.9 | −1.5% | 1.3% (0.5–2.1) | 4.9% | **99.9%** |
| Filtered BC | 45.0 | 0.7% | 1.1% | 1.9% | 98.4% |
| BC | 38.7 | 14.6% | 12.2% | 11.4% | 71.6% |

* **BC fails on comfort.** It saves energy only by under-conditioning. Because the logs contain random commissioning days and deadband days, imitating the average action produces causal confusion, for example sitting at 17 °C.
* **IQL is the best model-free method**, but its edge over the schedule is modest. The rule-based baseline is already near-optimal on comfort, and fine-grained Q-value differences (cents per step) are hard to resolve from 192k transitions.
* **MPC wins on cost by a wide margin.** It shifts load out of the 12–20 h peak window, pre-cools or pre-heats in off-peak hours, and flattens the daily peak (−26%), which is what demand charges reward.
* **Lagrangian CQL behaves as designed:** the highest comfort of any method, bought with some energy.

### 4.2 Robustness (cost saved / comfort)

| Scenario | MPC | IQL | CQL | LCQL | Rule comfort |
|---|---|---|---|---|---|
| Heat wave (+8 °C) | 2.6% / 100% | 2.0% / 97.1% | −0.2% / 96.7% | −2.7% / 96.8% | 100% |
| Cold snap (−10 °C) | 9.3% / 100% | 3.3% / 100% | 4.0% / 99.4% | −6.4% / 99.8% | 100% |
| Hot-arid climate | 4.6% / 99.6% | 2.1% / 94.6% | −0.8% / 91.8% | −3.5% / 92.0% | 100% |
| Volatile weather | 14.3% / 99.3% | 4.4% / 98.9% | 2.0% / 98.3% | −2.2% / 98.8% | 100% |
| Sensor noise (σ 0.5 °C) | 15.5% / 99.6% | 5.0% / 96.4% | −0.8% / 94.3% | −3.9% / 97.1% | 100% |
| Sensor drift (+1 °C) | 14.3% / 97.2% | 0.3% / 85.6% | −10.0% / 71.6% | −10.1% / 82.2% | 100% |
| 50% capacity loss | 13.2% / 91.7% | 5.3% / 91.8% | 3.2% / 91.0% | 3.4% / 93.1% | 93.4% |
| COP −30% | 11.4% / 98.8% | 5.5% / 99.4% | 3.3% / 99.0% | −0.9% / 99.4% | 100% |
| Boston (TMYx) | 13.1% / 99.3% | 5.6% / 97.0% | 2.2% / 98.1% | −2.0% / 98.2% | 100% |
| Chicago (TMYx) | 15.8% / 99.0% | 6.7% / 98.6% | 2.8% / 98.3% | −2.3% / 98.9% | 100% |
| Phoenix (TMYx) | 3.1% / 99.7% | 2.9% / 93.2% | −0.9% / 86.3% | −3.4% / 85.6% | 99.8% |
| Miami (TMYx) | 3.8% / 100% | 1.7% / 95.9% | −0.7% / 98.5% | −2.2% / 98.8% | 100% |

* MPC keeps a **10.1% average cost saving** across the 12 unseen scenarios. Its comfort stays ≥ 97% except under 50% capacity loss, where no policy, including the rule, can hold the band.
* Model-free policies degrade most in hot climates (Phoenix, hot-arid) and under sensor drift, which are the states least covered by the logs.
* **OOD fallback** triggers far more often out of distribution (0.4–0.7% of steps in-distribution vs. up to 9% under cold snap or Phoenix weather) but adds only ~0–4 points of comfort. Ensemble disagreement on dynamics detects novel *weather*, but it does not detect a policy's *value* error in familiar-looking states, for example drifted sensors.

### 4.3 Off-policy evaluation (35 candidates: all checkpoints, sweep variants, rule, MPC)

| Estimator | Spearman ρ with truth | MAE | Regret@1 | Picked |
|---|---|---|---|---|
| **FQE** | **0.82** | **1.05** | **0.02** | MPC (seed 1) |
| WDR | 0.58 | 3.91 | 0.65 | IQL (n4 sweep) |
| PDWIS | 0.33 | 3.59 | 15.48 | an LCQL sweep variant |

With 96-step episodes and deterministic targets, importance weights collapse (ESS ≈ 1). FQE is the
estimator to trust for choosing a policy from logs only, and it would have selected the actual best policy.

### 4.4 Dataset ablations (seed 0, in-distribution; cost saved / comfort)

| Dataset | MPC | IQL | CQL | LCQL |
|---|---|---|---|---|
| 250 days | 19.7% / 99.0% | 3.0% / 93.8% | 7.0% / 90.4% | 0.8% / 94.8% |
| 500 days | 17.6% / 99.7% | 9.8% / 97.5% | 3.1% / 97.7% | −4.7% / 97.4% |
| 1000 days | 17.4% / 99.3% | 3.8% / 100% | 4.0% / 99.4% | 2.6% / 98.8% |
| 2000 days (main) | 17.5% / 99.6% | 5.8% / 98.9% | −0.1% / 100% | −1.7% / 100% |
| Expert-only behavior | 22.8% / 99.4% | 2.8% / 100% | 2.8% / 98.5% | −0.2% / 100% |
| Random-heavy behavior | 17.5% / 99.2% | 4.4% / 99.6% | 2.1% / 98.4% | −4.7% / 97.6% |

* **Model-based learning is very data-efficient.** The dynamics are low-dimensional and smooth, and 250 days already identify them.
* Model-free results with small data are noisy and lose comfort (90–95% at 250 days).
* Single-seed rows vary by a few points, so treat differences under ~3 points as noise.

### 4.5 Multi-zone (3 zones, shared policy, 3 seeds, in-distribution)

| Policy | Energy saved | Cost saved | Peak cut | Comfort (S / N / core) |
|---|---|---|---|---|
| CQL | −2.2% | 4.0% | 16.4% | 97.1 / 98.8 / 99.0% |
| LCQL | −11.2% | 1.6% | 19.7% | 98.1 / 99.7 / 99.8% |

Parameter sharing scales to multiple zones and learns peak shaving (16–20% lower building peak). The
energy cost, however, makes the net gains small. The south perimeter zone, with its solar load, is the
hardest to keep in band.

### 4.6 Explainability

Permutation importance (accuracy drop when a feature is shuffled) and a depth-5 distilled decision tree:

| Policy | Top features | Tree fidelity |
|---|---|---|
| IQL | T_in, occupied, price_next2h, T_out | 96.6% |
| LCQL | T_in, price_next2h, occupied, hour | 92.3% |
| MPC | hour, T_out, T_in, solar, weekday | 88.2% |

The learned policies are close to small rule sets. For example, MPC pre-conditions to 20–21 °C during the 2 h
before occupancy when it is cold and the next-2 h price is low, and otherwise stays OFF while unoccupied.
These distilled trees are a practical deployment path, because a facility manager can read and audit them.
The rule files are `results/explain/*_rules.txt`.

## 5. Comparison with the original claims

The original project description claimed −18.5% energy, −21.4% cost and 99.2% comfort. These experiments
measure **−7.6% energy, −17.0% cost, −25.7% peak demand at 99.7% comfort (MPC)**, and −7.2% / −6.9% / 99.6% for
the best model-free method (IQL). The cost and comfort claims are approximately met. The energy figure is not,
mainly because the baseline schedule already sets back overnight, which leaves limited room for energy
(as opposed to cost) savings.

## 6. Limitations

* The building, dataset and evaluation are all simulated. The OOD scenarios are designed perturbations rather than field data.
* MPC assumes the occupancy schedule and tariff are known, plus an interpolated 3 h outdoor-temperature forecast.
* The EnergyPlus/Sinergym transfer adapter (`scripts/sinergym_transfer.py`) is written but has not been executed.
* Ablations use a single training seed (multi-zone uses 3).

## 7. Reproduce

`./scripts/run_all.sh` (≈ 3.5 h on a laptop, 10 parallel jobs) regenerates every number and figure in this report.
Versions are pinned in `requirements-lock.txt`.
