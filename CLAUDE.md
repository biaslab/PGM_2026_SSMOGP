# CLAUDE.md

## Project Summary

RxBayesOpt implements multi-output Bayesian optimization using state-space Gaussian Processes with inference via RxInfer.jl (reactive message passing).

## Language & Environment

- **Julia** (activate with `--project=.`)
- Dependencies in `Project.toml` — install with `Pkg.instantiate()`
- No test suite exists yet

## File Structure

```
Project.toml              — Julia package definition + dependencies
params.yaml               — Experiment hyperparameters (DVC-tracked)
dvc.yaml                  — DVC pipeline definition (1 stage)
experiments/
  partial_obs.jl          — Partial observations — FFG modularity (d=6, D=4)
  ett_forecast.jl         — ETTh1 oil-temperature forecasting (d=1, D=4)
experiments.jl            — Legacy: ad-hoc eval_blackbox, single SS-GP run
experiments_baseline.jl   — Legacy: baseline-only runner
experiments_comparison.jl — Legacy: multi-seed comparison
src/
  RxBayesOpt.jl           — Module definition, includes, and exports
  types.jl                — AbstractBOState, ExperimentConfig, POState, BOResult
  utils.jl                — row, sqdist, nn_chain_order, blockdiag
  statespace.jl           — matern32_blocks_from_Δ, additive_multioutput_blocks_from_Δ
  model.jl                — @model additive_gp_po (RxInfer model)
  acquisition.jl          — ucb_acquisition, select_next_point
  hyperparameters.jl      — log_marginal_likelihood, tune_hyperparameters
  visualization.jl        — plot_bo_step (3-panel BO visualization)
  bo.jl                   — setup_experiment, _current_best, save_results
  baseline.jl             — BaselineState, setup_baseline, run_bo_baseline! (LMC kernel-matrix GP)
  partial_obs.jl          — POState, BaselinePOState, setup_po, run_bo_po!, partial-obs baseline, run_po_comparison
  benchmarks.jl           — Standard benchmark functions (Hartmann-6, environmental model)
  data_ett.jl             — load_etth1 (reads data/ett/ETTh1.csv, subsamples N timestamps)
  forecasting.jl          — setup_forecast, run_forecast_po, run_forecast_baseline_po, run_ett_comparison
```

## Code Map

### types.jl
- `AbstractBOState` — abstract supertype for BO state objects (subtypes: BaselineState, POState, BaselinePOState)
- `ExperimentConfig` — immutable `@kwdef` struct: N, d, Q, D, ℓs, σ2s, β, s, n_seed, steps, tune_every, R_diag_init, animate, log_every, seed, obs_pattern, obs_frac
- `POState <: AbstractBOState` — mutable struct: blocks, W, τ, e_vecs, mask, Y, Y_flat, μy, σy (partial-observation state)
- `BOResult` — immutable struct: best_index, best_value, best_y, observed_indices, n_iterations, R_learned, best_value_history, n_observed_history, R_diag_history, step_times, method
- `print_config(cfg)` / `print_summary(result)` — logging helpers

### utils.jl
- `row(X, i)` — zero-allocation row view
- `sqdist(a, b)` — fast squared Euclidean distance (SIMD)
- `nn_chain_order(X)` — greedy nearest-neighbor chain ordering
- `blockdiag(mats...)` — manual block-diagonal matrix construction

### statespace.jl
- `matern32_blocks_from_Δ(Δ; ℓ, σ2)` — Discretizes Matérn 3/2 SDE into state-space matrices (A, Q, P∞, H)
- `additive_multioutput_blocks_from_Δ(Δ; ℓs, σ2s, W)` — Stacks Q latent GPs into block-diagonal state-space model with mixing matrix W

### model.jl
- `additive_gp_po` — `@model` macro: per-output scalar noise factors for partial observations

### acquisition.jl
- `ucb_acquisition(res, state, cfg, N)` — computes UCB from RxInfer posteriors, returns (ucb, μ_pred, σ_pred, μs, σs)
- `select_next_point(ucb, Y)` — returns unobserved point with highest UCB

### hyperparameters.jl
- `log_marginal_likelihood(Y_obs, P, A, Q, H, R)` — Kalman filter forward pass
- `tune_hyperparameters(Y_obs, Δ, W_init, D, Q)` — LBFGS Type-II ML optimization

### visualization.jl
- `plot_bo_step(step, k, state, acq, Ytrue, cfg)` — 3-panel plot (scalarized prediction, UCB, first 3 outputs)

### bo.jl
- `setup_experiment(cfg, eval_fn)` — generates data, computes NN chain, builds SS-GP blocks; returns `(; Xo, Δ, blocks, W, Y, μy, σy, Ytrue, dist_norm)`
- `_current_best(state, cfg)` — finds best observed point by scalarized value
- `save_results(result, cfg, frames)` — saves metrics JSON, PNG, GIF

### baseline.jl
- `BaselineState <: AbstractBOState` — mutable struct: W, ℓs, σ2s, R, dist_norm, Xo, Y, μy, σy
- `setup_baseline(cfg, setup_data)` — creates BaselineState from setup_experiment output (same W, ℓs, σ2s; fixed R)
- `_lmc_kernel_matrix(X, W, ℓs, σ2s, R, dist_norm)` — builds full (D*n × D*n) LMC covariance matrix
- `_lmc_predict(bl_state)` — LMC kernel-matrix GP prediction via Cholesky
- `baseline_ucb_acquisition(bl_state, cfg)` — UCB from LMC GP predictions
- `run_bo_baseline!(cfg, eval_fn; bl_state, Ytrue)` — BO loop with LMC kernel-matrix GP

### partial_obs.jl
- `POState <: AbstractBOState` — mutable struct for partial-observation SS-GP BO
- `BaselinePOState <: AbstractBOState` — mutable struct for partial-observation KM-GP BO
- `_generate_obs_mask(cfg, N, rng)` — generates N × D observation mask (:full or :sensor_groups)
- `setup_po(cfg, setup_data)` — creates POState from setup_experiment output
- `run_bo_po!(cfg, eval_fn; po_state, Xo, Δ, Ytrue)` — BO loop with partial-observation SS-GP
- `setup_baseline_po(cfg, setup_data, mask)` — creates BaselinePOState
- `_lmc_predict_po(bl_state)` — kernel-matrix GP with variable-size covariance for partial obs
- `run_bo_baseline_po!(cfg, eval_fn; bl_state, Ytrue)` — BO loop with partial-observation KM-GP
- `run_po_comparison(cfg_template, eval_fn; seeds, output_dir)` — 4-way comparison: SS/KM × full/partial

### benchmarks.jl
- `hartmann6(x)` — Standard Hartmann 6-dimensional function on [0,1]^6, global max ≈ 3.3224
- `make_mo_hartmann(; D, Q, W_seed)` — Factory: multi-output Hartmann via LMC mixing of permuted Hartmann-6 latents
- `make_environmental(; n_spatial, n_temporal)` — Factory: environmental monitoring model (Bliznyuk et al. 2008, Maddox et al. NeurIPS 2021)

### experiments/partial_obs.jl
- Partial observation on environmental monitoring benchmark (d=4, D=12, N=500, 5 seeds)
- 4-way: SS-full, SS-PO, KM-full, KM-PO — demonstrates FFG modularity

### data_ett.jl
- `load_etth1(; N, csv_path, cols)` — reads `data/ett/ETTh1.csv`, evenly subsamples N rows, extracts the requested columns. Returns `(X, Y, col_names, ot_idx)` where X is N×1 normalized time on [-1, 1].

### forecasting.jl
- `_generate_forecast_mask(N, D, train_frac, dropout, rng)` — mask is true on training half with prob `1-dropout`, false on test half
- `setup_forecast(cfg, X, Y_data, mask)` — natural 1D ordering, per-output standardization from training-observed entries only (no test leakage)
- `setup_forecast_po(cfg, setup_data)` / `setup_forecast_baseline_po(cfg, setup_data)` — variants of setup_po / setup_baseline_po that reuse the mask from setup_data
- `run_forecast_po(cfg, setup_data)` — single SS-GP inference call → predictions in original scale
- `run_forecast_baseline_po(cfg, setup_data)` — single KM-GP inference call
- `_compute_test_mse_nll(μ_pred, σ_pred, Ytrue, mask, d_target)` — MSE and NLL restricted to test rows for a specific output index
- `run_ett_comparison(cfg, X, Y_data; seeds, dropouts, train_frac, d_target, output_dir)` — sweep, save `comparison.json` + MSE/NLL plots

### experiments/ett_forecast.jl
- ETTh1 forecasting (d=1, D=4, N=500), 3 dropout levels × 5 seeds × 2 methods
- One-shot regression (no acquisition loop); metrics are MSE and NLL on OT

### Legacy scripts
- `experiments.jl` — ad-hoc eval_blackbox, single SS-GP run
- `experiments_baseline.jl` — baseline-only runner
- `experiments_comparison.jl` — multi-seed comparison with ad-hoc function

## Key Design Decisions
- **State-space GP instead of kernel-matrix GP** — O(N) vs O(N³) scaling
- **NN chain ordering** — heuristic to enable state-space GP on high-dim inputs
- **Additive latent structure** — Q independent latent GPs mixed by W (LMC)
- **RxInfer for inference** — automated message passing, potential for reactive/online updates
- **Per-output scalar noise** — `NormalMeanPrecision` per output dim; naturally handles partial observations via message passing
- **UCB with scalarization** — multi-output → scalar via preference vector s
- **No globals** — all mutable state in POState/BaselinePOState structs
- **Hyperparameter tuning** — controlled by `cfg.tune_every` (0 = disabled)

## DVC Pipeline

Run all experiments: `dvc repro`
Run one experiment: `dvc repro partial_obs`
View metrics: `dvc metrics show`

Configuration is in `params.yaml`. Changes to params or source code trigger re-runs via DVC.

### Stages
1. **partial_obs** — Partial observation 4-way comparison on environmental benchmark (d=4, D=12, N=500, 5 seeds). Outputs: `data/partial_obs/`
2. **ett_forecast** — ETTh1 forecasting (d=1, D=4, N=500), MSE & NLL on OT vs dropout. Needs `data/ett/ETTh1.csv` checked in or downloaded manually. Outputs: `data/ett_forecast/`

## Known Issues / Improvement Opportunities
- No test suite
- The NN chain ordering is a greedy heuristic; quality depends on starting point
- `blockdiag` is hand-rolled instead of using BlockDiagonals.jl
- Seed observations are randomly selected rather than space-filling (e.g., LHS)
- No convergence criterion — always runs for a fixed number of steps
- Scalarization vector `s` is fixed; could be adaptive or user-specified

## Conventions
- Julia naming: snake_case for functions, PascalCase for types
- Greek letters used in variable names (ℓ, σ2, Δ, μ, β) — common in probabilistic ML code
- `@inline` used on hot utility functions
- All public functions have docstrings
