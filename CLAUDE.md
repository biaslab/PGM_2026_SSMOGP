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
  partial_obs.jl          — ETTh multi-dim-input regression under feature dropout: SS-LMC vs KM-LMC vs SVGP-LMC
  dim_sweep.jl            — Input-dimension sweep on synthetic sensor network (SS vs KM vs Random)
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
  partial_obs.jl          — POState, BaselinePOState, setup_po, run_bo_po!, cov-restructuring KM-LMC partial-obs baseline
  dim_sweep.jl            — run_dim_sweep, random-acquisition baseline, dim-sweep plots
  benchmarks.jl           — Standard benchmark functions (Hartmann-6, environmental model, sensor network)
  ett.jl                  — ETTh multi-dim-input regression under feature dropout: SS-LMC (RxInfer + raw Kalman) vs KM-LMC vs SVGP-LMC
  ss_lmc_raw.jl           — Hand-coded Kalman filter + RTS smoother for the additive multi-output state-space LMC (no RxInfer)
  vecchia.jl              — Vecchia/NNGP LMC baseline (maxmin ordering, m nearest-neighbour conditioning sets)
test/
  test_vecchia.jl         — Correctness tests for the Vecchia baseline
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
- `nn_chain_quality(Xo)` — diagnostic summary of consecutive-chain distances (mean/median/max/min/total + raw Δ)
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
- `setup_baseline_po(cfg, setup_data, mask)` — creates BaselinePOState; precomputes `K_prior_full` (full noise-free D·N×D·N LMC kernel)
- `_lmc_full_kernel(Xo, W, ℓs, σ2s, dist_norm)` — builds the full structured LMC kernel (point-major flat indexing)
- `_lmc_predict_po(bl_state)` — KM-LMC prediction by **covariance restructuring**: slices the observed sub-block out of `K_prior_full`, factorizes, predicts
- `run_bo_baseline_po!(cfg, eval_fn; bl_state, Ytrue)` — BO loop with partial-observation KM-GP (still used by dim_sweep/sequential_design)

### ett.jl
- `load_ett(path; n_rows)` — manual CSV loader for ETTh1 (drops `date`, returns n×7 matrix; column order HUFL,HULL,MUFL,MULL,LUFL,LULL,OT)
- `_ett_setup(data, N_train, N_test, p, seed, D, Q, input_cols, output_cols)` — slices ETTh into 3D inputs (useful loads) and 4D outputs (useless loads + OT), standardizes by training-half stats, NN-chain orders all 2N points jointly, applies per-cell dropout to training-half outputs only, returns `(Xo, Δ, dist_norm, W, mask, Y, Y_flat, Ytrue, test_idx_chain, …)`
- `forecast_ss(setup, …)` — SS-LMC forecast via `additive_gp_po` message passing along the NN chain → (mnll, rmse, time)
- `forecast_ss_raw(setup, …)` — SS-LMC forecast via the hand-coded Kalman filter + RTS smoother in `ss_lmc_raw.jl` (same SS blocks; no RxInfer)
- `forecast_km(setup, …)` — KM-LMC forecast via cov restructuring on the full (D·N)² LMC kernel over chain-ordered inputs
- `forecast_svgp(setup, …, M; Z_seed)` — SVGP-LMC forecast; K-means inducing points drawn from training chain positions only
- `_forecast_mnll` / `_test_rmse` — metrics over the chain-position test indices `setup.test_idx_chain`
- `run_ett_forecast(data, N, p, seed; D, Q, ℓs, σ2s, R_diag_init, input_cols, output_cols, M)` — one train(first N rows)/forecast(next N rows) comparison; consumes 2·N rows total
- `run_ett_sweeps(data; Ns, ps, N_fixed, N_fixed_big, p_fixed, seeds, …)` — three sweeps: N (at `p_fixed`), p (at `N_fixed`), and p at the high-C window `N_fixed_big`; saves comparison.json + time/MNLL/RMSE plots (with `_at_Cbig` suffix for the high-C p-sweep)

### benchmarks.jl
- `hartmann6(x)` — Standard Hartmann 6-dimensional function on [0,1]^6, global max ≈ 3.3224
- `make_mo_hartmann(; D, Q, W_seed)` — Factory: multi-output Hartmann via LMC mixing of permuted Hartmann-6 latents
- `make_environmental(; n_spatial, n_temporal)` — Factory: environmental monitoring model (Bliznyuk et al. 2008, Maddox et al. NeurIPS 2021)
- `make_synthetic_1d(; D, Q, W_seed)` — 1D sinusoidal multi-output benchmark (NN chain is exact)
- `make_sensor_network(; d, D, station_seed, jitter_seed, σ_spatial)` — Synthetic d-parametric sensor-network benchmark: each input dim is one weather station, outputs are spatially-weighted saturating combinations

### dim_sweep.jl
- `_run_random_bo(cfg, eval_fn; Xo, setup_data)` — Random-acquisition baseline (uniform pick over unobserved)
- `run_dim_sweep(cfg_template, eval_fn_factory; ds, seeds, output_dir)` — Sweep over input dim, compare SS-GP / KM-GP / Random and log NN-chain quality per (d, seed)
- `_plot_dim_sweep(results, ds, output_dir)` — Per-d convergence + final-regret/time/chain-quality vs d

### experiments/partial_obs.jl
- ETTh multi-dim-input regression under random per-feature dropout. Inputs: HUFL,MUFL,LUFL (d=3). Outputs: HULL,MULL,LULL,OT (D=4, Q=3). First N rows train (with dropout on outputs); next N rows fully held out. NN-chain ordering on the 3D inputs lets SS-LMC handle multi-dim.
- 3-way: SS-LMC (message passing) vs KM-LMC (cov restructuring) vs SVGP-LMC (inducing pts), swept over window C (=N), dropout p, and a second p-sweep at a higher C (`N_fixed_big`)
- Demonstrates SS-LMC matches KM-LMC accuracy on real multi-dim data while being faster at large N (O(N) vs O((D·N)³))

### experiments/dim_sweep.jl
- Input-dimension sweep on synthetic sensor network (d ∈ {2,4,8,16,32}, D=3, N=200, 5 seeds)
- 3-way: SS-GP vs KM-GP vs Random — quantifies NN-chain degradation with d

### Legacy scripts
- `experiments.jl` — ad-hoc eval_blackbox, single SS-GP run
- `experiments_baseline.jl` — baseline-only runner
- `experiments_comparison.jl` — multi-seed comparison with ad-hoc function

### ss_lmc_raw.jl
- `ss_lmc_filter_smooth(P, A, Q, H, τ, Y_flat, N, D)` — Kalman filter (sequential scalar updates with Joseph-form covariance) + RTS smoother for the additive multi-output state-space LMC. Returns per-chain-position posterior `(μ_pred, σ_pred)` on `H·f`. Same SS blocks as `additive_gp_po`; bypasses RxInfer.

## Key Design Decisions
- **Two SS-LMC implementations** — `forecast_ss` calls RxInfer's `additive_gp_po` (automated message passing, user-facing); `forecast_ss_raw` runs a hand-coded Kalman filter + RTS smoother (`ss_lmc_filter_smooth`). The raw version is used for fair scalability comparison against KM-LMC and SVGP-LMC, and as a numerical diagnostic.
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
1. **partial_obs** — ETTh multi-dim-input regression under feature dropout (3D input from useful loads, 4D output of useless loads + OT); SS-LMC vs KM-LMC vs SVGP-LMC. Sweeps over N, p (at low C), and p (at high C). Outputs: `data/partial_obs/`
2. **dim_sweep** — Input-dimension sweep on synthetic sensor network (d ∈ {2,4,8,16,32}, D=3, N=200, 5 seeds). Outputs: `data/dim_sweep/`

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
