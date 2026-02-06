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
experiments.jl            — Entry point: defines eval_blackbox, config, runs BO
src/
  RxBayesOpt.jl           — Module definition, includes, and exports
  types.jl                — ExperimentConfig, BOState, BOResult structs
  utils.jl                — row, sqdist, nn_chain_order, blockdiag
  statespace.jl           — matern32_blocks_from_Δ, additive_multioutput_blocks_from_Δ
  model.jl                — @model additive_gp_vv (RxInfer model)
  acquisition.jl          — ucb_acquisition, select_next_point
  hyperparameters.jl      — log_marginal_likelihood, tune_hyperparameters
  visualization.jl        — plot_bo_step (3-panel BO visualization)
  bo.jl                   — setup_experiment, run_bo!, _current_best
```

## Code Map

### types.jl
- `ExperimentConfig` — immutable `@kwdef` struct: N, d, Q, D, ℓs, σ2s, β, s, n_seed, steps, tune_every, R_diag_init, animate, log_every, seed
- `BOState` — mutable struct: blocks, W, R, Y, μy, σy (evolving state during BO)
- `BOResult` — immutable struct: best_index, best_value, best_y, observed_indices, n_iterations
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
- `additive_gp_vv` — `@model` macro: linear dynamical system for RxInfer message passing

### acquisition.jl
- `ucb_acquisition(res, state, cfg, N)` — computes UCB from RxInfer posteriors, returns (ucb, μ_pred, σ_pred, μs, σs)
- `select_next_point(ucb, Y)` — returns unobserved point with highest UCB

### hyperparameters.jl
- `log_marginal_likelihood(Y_obs, P, A, Q, H, R)` — Kalman filter forward pass
- `tune_hyperparameters(Y_obs, Δ, W_init, D, Q)` — LBFGS Type-II ML optimization

### visualization.jl
- `plot_bo_step(step, k, state, acq, Ytrue, cfg)` — 3-panel plot (scalarized prediction, UCB, first 3 outputs)

### bo.jl
- `setup_experiment(cfg, eval_fn)` — generates data, computes NN chain, initializes BOState
- `run_bo!(cfg, eval_fn; Xo, Δ, state, Ytrue)` — main BO loop, returns BOResult
- `_current_best(state, cfg)` — finds best observed point by scalarized value

### experiments.jl
- Defines `eval_blackbox` (synthetic 10-output function)
- Creates `ExperimentConfig` with defaults, runs `setup_experiment` + `run_bo!`

## Key Design Decisions
- **State-space GP instead of kernel-matrix GP** — O(N) vs O(N³) scaling
- **NN chain ordering** — heuristic to enable state-space GP on high-dim inputs
- **Additive latent structure** — Q independent latent GPs mixed by W (LMC)
- **RxInfer for inference** — automated message passing, potential for reactive/online updates
- **UCB with scalarization** — multi-output → scalar via preference vector s
- **No globals** — all mutable state in BOState struct
- **Hyperparameter tuning** — controlled by `cfg.tune_every` (0 = disabled)

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
