---
name: rxbayesopt-method
description: Method and mathematical details for the paper "Scalable Multi-Output Bayesian Optimization via State-Space GPs and Reactive Message Passing". Use when writing, editing, or reviewing sections of the paper, generating LaTeX equations, creating figures, writing the abstract, or discussing the method.
user-invocable: false
---

# Scalable Multi-Output Bayesian Optimization via State-Space GPs and Reactive Message Passing

## Paper Scope

This paper presents a Bayesian optimization (BO) framework for expensive black-box functions with multiple correlated outputs. The key idea is to cast the surrogate GP model as a state-space model (linear dynamical system) and perform inference via variational message passing (VMP) in a factor graph using RxInfer.jl. The observation noise precision is learned online from a Wishart prior, making the model adaptive. A greedy nearest-neighbor chain ordering heuristic enables the state-space formulation on high-dimensional inputs. The method is compared against a kernel-matrix GP baseline using the same LMC model structure but with fixed observation noise.

## Contributions

1. **State-space GP surrogate for BO** -- Replace the O(N^3) kernel-matrix GP with an O(N) state-space formulation based on the Matern 3/2 SDE, enabling scalable surrogate modeling.
2. **Nearest-neighbor chain ordering** -- A greedy heuristic that maps high-dimensional inputs onto a 1D chain preserving local distance structure, making state-space GPs applicable beyond 1D inputs.
3. **Multi-output state-space LMC** -- The Linear Model of Coregionalization (LMC) is realized in state-space form with block-diagonal transition and noise matrices and a mixing observation matrix, retaining O(N) cost.
4. **Online noise learning** -- The observation noise precision Lambda is given a Wishart prior and learned online via variational message passing. The posterior from each BO step becomes the prior for the next, enabling the model to adapt its noise estimate as data accumulates.
5. **Declarative probabilistic model** -- The entire model is specified as a probabilistic program in RxInfer's `@model` DSL (~10 lines), and inference is automated message passing on the factor graph. This contrasts with hand-coded Kalman filters and enables reactive/streaming extensions.

## Mathematical Formulation

### Matern 3/2 State-Space Representation

A scalar GP with Matern 3/2 covariance function and length-scale ell, signal variance sigma^2, can be represented exactly as the solution of the stochastic differential equation (SDE):

```
dx(t) = F x(t) dt + L dw(t)
```

where x(t) = [f(t), f'(t)]^T is the 2D state, and:

```
F = [0, 1; -lambda^2, -2*lambda],   lambda = sqrt(3) / ell
```

The stationary covariance is:

```
P_inf = [sigma^2, 0; 0, lambda^2 * sigma^2]
```

Given inter-point distance Delta_i between consecutive points, the discrete-time transition and process noise matrices are:

```
A_i = exp(F * Delta_i)
Q_i = P_inf - A_i * P_inf * A_i^T
```

The observation model extracts the function value from the state:

```
h = [1, 0]^T,   so   f_i = h^T x_i
```

This gives a linear-Gaussian state-space model equivalent to the GP, with O(N) inference via the Kalman filter/smoother.

**Implementation**: `matern32_blocks_from_Δ(Δ; ℓ, σ2)` in `src/statespace.jl` computes A, Q, P_inf, H for all N points. Process noise Q_i is clamped to PSD via eigendecomposition for numerical stability.

### Nearest-Neighbor Chain Ordering

State-space GPs assume a 1D ordering of data points. For d-dimensional inputs X in R^{N x d}, we construct a 1D chain via greedy nearest-neighbor ordering:

1. Start from the first point (row 1)
2. At each step, visit the closest unvisited point (squared Euclidean distance)
3. Record the inter-point distances Delta_i = ||x_{pi(i)} - x_{pi(i-1)}||

The distances Delta are normalized by their median: Delta <- Delta / median(Delta). This normalization factor (`dist_norm`) is stored so the kernel-matrix baseline can normalize its pairwise distances identically. The normalized distances serve as the "time steps" in the state-space model: nearby points in the original space have small Delta (tight coupling) and distant points have large Delta (weak coupling, approaching the prior).

**Approximation implications**: The chain ordering projects high-dimensional structure onto a 1D sequence. This introduces three layers of approximation vs the exact kernel matrix:
1. Chain ordering: only adjacent-point correlations are modeled directly; long-range correlations are mediated through the chain.
2. Markov structure: the state-space model is Markov (each state depends only on the previous), losing some multi-point correlation structure.
3. Path dependence: the greedy heuristic is not globally optimal; starting from a different point yields a different chain. The implementation always starts from index 1.

**Implementation**: `nn_chain_order(X)` in `src/utils.jl`.

### Multi-Output: Linear Model of Coregionalization (LMC)

For D output dimensions, we use Q independent latent GPs, each with its own length-scale ell_q and variance sigma^2_q. The latent GPs are combined through a mixing matrix W in R^{D x Q}:

```
y_i = W * [f_i^{(1)}, ..., f_i^{(Q)}]^T + epsilon_i
```

In state-space form, the full state x_i in R^{2Q} is the concatenation of all latent states. The system matrices are block-diagonal:

```
A_i = blkdiag(A_i^{(1)}, ..., A_i^{(Q)})       [2Q x 2Q]
Q_i = blkdiag(Q_i^{(1)}, ..., Q_i^{(Q)})       [2Q x 2Q]
P   = blkdiag(P_inf^{(1)}, ..., P_inf^{(Q)})   [2Q x 2Q]
```

The observation matrix combines the mixing matrix W with the per-GP observation vectors:

```
H = [W[:,1] * h^{(1)T}, ..., W[:,Q] * h^{(Q)T}]   [D x 2Q]
```

so that H * x_i gives a D-dimensional output. This preserves O(N*Q) scaling.

**Implementation**: `additive_multioutput_blocks_from_Δ(Δ; ℓs, σ2s, W)` in `src/statespace.jl`.

### Probabilistic Model (State-Space GP)

The full generative model, expressed as a factor graph for message passing:

```
Lambda ~ Wishart(nu_0, S_0)                    (observation precision prior)
x_0 ~ N(0, P)                                  (stationary initial state)
x_i ~ N(A_i * x_{i-1}, Q_i)   for i=1..N      (state transition)
m_i = H * x_i                                  (deterministic observation mean)
y_i ~ N(m_i, Lambda^{-1})     for i=1..N      (observation likelihood)
```

where Lambda is the D x D observation **precision** matrix (inverse of the noise covariance R). It is given a Wishart prior initialized so that the expected noise covariance is R_diag_init * I_D:

```
nu_0 = D + 2
S_0 = I_D / (nu_0 * R_diag_init)
=> E[Lambda] = nu_0 * S_0 = I_D / R_diag_init
=> E[R] = E[Lambda^{-1}] ≈ R_diag_init * I_D
```

Because Lambda is a random variable (not fixed data), inference requires **variational message passing (VMP)** with a mean-field factorization `q(Lambda, m) = q(Lambda) q(m)`. The model is no longer purely linear-Gaussian, so a single Kalman pass is insufficient.

**Online noise learning**: After each BO step, the Wishart posterior on Lambda replaces the prior for the next step: `state.R_prior = last(res.posteriors[:Lambda])`. This creates a sequential Bayesian update where the noise estimate improves as more observations arrive. The posterior can learn a **full (non-diagonal)** precision matrix, capturing cross-output noise correlations.

Missing observations (y_i = missing) are handled natively by RxInfer's message passing -- predictions flow through the graph without an observation update.

**Implementation**: `@model function additive_gp_vv(Y, P, A, Q, H, prior_Λ)` in `src/model.jl`. Note: RxInfer's `@model` macro cannot have Julia docstrings attached; `#` comments are used instead.

### Inference: Variational Message Passing

Because the observation precision Lambda is a random variable with a Wishart prior, inference uses variational message passing (VMP) rather than an exact Kalman smoother:

```julia
res = infer(
    model=additive_gp_vv(P=blocks.P, A=blocks.A, Q=blocks.Q, H=blocks.H, prior_Λ=state.R_prior),
    data=(Y=state.Y,),
    iterations=10,
    initialization=@initialization(q(Λ) = state.R_prior),
    constraints=@constraints(q(Λ, my) = q(Λ)q(my)),
    predictvars=(Y=KeepLast(),),
    options=(limit_stack_depth=1000,)
)
```

Key details:
- **10 VMP iterations** per BO step for convergence of the variational posterior
- **Mean-field factorization** `q(Lambda, m) = q(Lambda) q(m)` -- this separates the noise precision from the observation means, enabling Wishart-Normal conjugate updates
- **Initialization** sets `q(Lambda)` to the current Wishart belief (prior or posterior from previous step)
- **Predictions** at all N points (observed and unobserved) are available via `predictvars`
- After inference, `state.R_prior = last(res.posteriors[:Λ])` updates the noise belief for the next step

The posteriors for the observation means `m_i = H * x_i` provide the predictive distributions at all N points. At observed points, predictions are conditioned on the data; at unobserved points, predictions propagate uncertainty through the chain.

**Per-step cost**: O(N * (2Q)^3 * n_iter) where n_iter=10. Since N, Q, and n_iter are fixed, this is O(N) per step -- it does not grow with the number of observed points n_obs.

### Kernel-Matrix GP Baseline

The baseline uses the same LMC model structure (same W, ℓs, σ2s) but differs in two ways:
1. **Inference method**: Exact kernel matrix + Cholesky decomposition (O((D*n_obs)^3)) instead of O(N) message passing
2. **Fixed observation noise**: R = R_diag_init * I(D) is constant throughout, never updated

The baseline performs:

1. Build the full (D*n_obs x D*n_obs) LMC covariance matrix:
   ```
   K = sum_q kron(W[:,q]*W[:,q]', K_q) + kron(R, I(n_obs))
   ```
   where K_q[i,j] = matern32(||x_i - x_j|| / dist_norm, ell_q, sigma^2_q)

2. Cholesky factorize: L = chol(K)  (with 1e-8 jitter for stability)
3. Solve: alpha = K^{-1} y_obs
4. For each candidate point, compute cross-covariance and predictive mean/variance

Pairwise Euclidean distances are normalized by the same `dist_norm` (median of chain distances) used by the state-space method, ensuring that length-scales have identical meaning in both methods.

**Per-step cost**: O((D*n_obs)^3) for Cholesky, growing cubically as observations accumulate. Additionally O(N * (D*n_obs)^2) for predictions at all N candidates.

**Implementation**: `_lmc_kernel_matrix`, `_lmc_predict`, `run_bo_baseline!` in `src/baseline.jl`. The `BaselineState` stores Xo, W, ℓs, σ2s, R (fixed matrix), dist_norm, Y, μy, σy.

### Differences Between SS-GP and KM-GP

The two methods share the same LMC model structure (same W, ℓs, σ2s, same initial data), but differ in:

| Aspect | SS-GP (State-Space) | KM-GP (Kernel-Matrix) |
|--------|---------------------|----------------------|
| Inference method | VMP on factor graph (10 iterations) | Exact Cholesky |
| Per-step scaling | O(N) (constant in n_obs) | O((D*n_obs)^3) |
| Observation noise R | Learned online via Wishart prior on Lambda | Fixed: R_diag_init * I(D) |
| Correlation structure | 1D chain (Markov, via NN ordering) | Full pairwise kernel matrix |
| Noise model flexibility | Full D x D precision (can learn cross-output noise) | Diagonal, isotropic |

The comparison thus tests **two differences simultaneously**: (1) the inference method/scaling and (2) adaptive vs fixed noise. Both methods use the same structural hyperparameters (W, ℓs, σ2s) which are fixed throughout all experiments.

### UCB Acquisition with Scalarization

Multi-output predictions are scalarized via a preference vector s in R^D (default: uniform s = [1/D, ..., 1/D]):

```
mu_s(x) = s^T * mu(x)
sigma_s(x) = sqrt(sum_j (s_j * sigma_j(x))^2)
```

where mu(x) and sigma(x) are the predicted mean and standard deviation vectors (original scale, after un-standardizing). The scalarized uncertainty assumes **independent output dimensions** in the predictive marginal (sums s_j^2 * sigma_j^2 rather than computing s^T * Sigma * s). The acquisition function is Upper Confidence Bound:

```
UCB(x) = mu_s(x) + beta * sigma_s(x)
```

At each step, the unobserved point with highest UCB is selected for evaluation. The exploration-exploitation tradeoff is controlled by beta (default 2.0).

**Implementation**: `ucb_acquisition` and `select_next_point` in `src/acquisition.jl`; `baseline_ucb_acquisition` in `src/baseline.jl` (same formula, different prediction backend).

### Hyperparameter Tuning (Optional -- Currently Disabled)

An implementation exists for tuning kernel hyperparameters via Type-II maximum likelihood using LBFGS (`tune_hyperparameters` in `src/hyperparameters.jl`). However, `tune_every=0` by default in `ExperimentConfig` and **no experiment overrides this**, so tuning is **never executed** in any current experiment. Both methods use the fixed hyperparameters (ℓs, σ2s, W) specified in `params.yaml` and generated in `setup_experiment`.

## Benchmark Functions

### Multi-Output Hartmann-6 (`make_mo_hartmann`)

Standard Hartmann 6-dimensional function (Dixon & Szego, 1978) extended to multiple outputs via LMC-style mixing. Used in the convergence and scaling experiments.

- **Inputs**: d=6, standardized ~N(0,1), mapped internally to [0,1]^6 via sigmoid
- **Outputs**: D configurable (default D=4), produced by mixing Q latent Hartmann-6 evaluations on permuted inputs through a fixed mixing matrix W_true
- **Latent functions**: Q <= 6 permuted versions of Hartmann-6 (fixed permutations in `_HARTMANN_PERMS`)
- **True mixing matrix**: W_true ~ N(0, 0.3) with added 0.7 on cyclic diagonal (output d loads on latent (d-1) % Q + 1), seeded with W_seed=42
- **Known optimum**: Single-output Hartmann-6 has global max ~3.3224
- **Correlation structure**: Genuine LMC-like correlations through W_true, making it a natural test for multi-output methods

Note: The BO methods use their **own random W** (from `randn(D,Q)*0.5` seeded by `cfg.seed`), not the true W_true. This is a proper blind optimization setup -- the model's mixing matrix is not the ground-truth one.

```julia
eval_fn = make_mo_hartmann(D=4, Q=3)  # returns a function x -> Vector{Float64}(length D)
```

**Implementation**: `hartmann6` and `make_mo_hartmann` in `src/benchmarks.jl`.

### Environmental Monitoring Model (`make_environmental`)

Pollutant diffusion benchmark from Bliznyuk et al. (2008) and Maddox et al. (NeurIPS 2021). Used in the high-dimensional output experiment.

- **Inputs**: d=4 physical parameters (source mass M, diffusivity Dc, source location L, decay time tau), standardized ~N(0,1), mapped to physical ranges via sigmoid
- **Outputs**: D = n_spatial * n_temporal concentration measurements (default 3*4 = 12), ordered time-major (inner loop spatial, outer loop temporal)
- **Physics**: Point-source pollutant in 1D medium with diffusion and exponential decay, including a reflecting boundary:
  ```
  c(s,t) = M/sqrt(4*pi*Dc*t) * [exp(-(s-L)^2/(4*Dc*t)) + exp(-(s+L)^2/(4*Dc*t))] * exp(-t/tau)
  ```
- **Correlation structure**: Natural spatial-temporal correlations through shared physics

```julia
eval_fn = make_environmental(n_spatial=3, n_temporal=4)  # d=4 -> D=12
```

**Implementation**: `make_environmental` in `src/benchmarks.jl`.

## Experiments

All experiments are managed via a DVC pipeline (`dvc.yaml`) with hyperparameters in `params.yaml`. Run all experiments with `dvc repro` or individual stages with `dvc repro <stage>`.

### Experiment 1: Convergence (`experiments/convergence.jl`)

**Goal**: Compare convergence quality of SS-GP (with online noise learning) vs KM-GP (fixed noise) on a standard benchmark.

| Parameter | Value |
|-----------|-------|
| Function | Multi-output Hartmann-6 |
| N | 300 |
| d | 6 |
| D | 4 |
| Q | 3 |
| Steps | 100 |
| Seeds | 10 (0-9) |
| n_seed | 10 |
| ℓs | [1.0, 2.0, 3.5] |
| σ2s | [3.0, 3.0, 3.0] |
| R_diag_init | 0.2 |
| beta | 2.0 |
| tune_every | 0 (disabled) |

**Outputs** (`data/convergence/`):
- `comparison.json`: per-seed, per-method best_value_history and step_times
- `convergence.png` + `.tikz`: mean +/- std convergence curves (both methods)
- `timing.png` + `.tikz`: median per-step timing comparison
- `metrics.json`: DVC metrics (mean final best values and total times)

### Experiment 2: Computational Scaling (`experiments/scaling.jl`)

**Goal**: Demonstrate O(N) vs O(n_obs^3) per-step scaling.

| Parameter | Value |
|-----------|-------|
| Function | Multi-output Hartmann-6 |
| N | 1000 |
| d | 6 |
| D | 4 |
| Q | 3 |
| Steps | 200 |
| Seeds | 5 (0-4) |
| n_seed | 10 |
| ℓs | [1.0, 2.0, 3.5] |
| σ2s | [3.0, 3.0, 3.0] |
| R_diag_init | 0.2 |
| beta | 2.0 |
| tune_every | 0 (disabled) |

**Outputs** (`data/scaling/`):
- `comparison.json`, `convergence.png`, `timing.png`: standard comparison outputs
- `scaling_perstep.png`: per-step time vs n_obs with O(1) and O(n^3) reference lines (log-scale y-axis). Note: TikZ export is disabled for log-scale plots.
- `scaling_cumulative.png`: cumulative wall-clock time vs n_obs
- `metrics.json`: DVC metrics (median step times, total times, KM/SS time ratio)

**Expected result**: SS-GP per-step time is flat (constant); KM-GP per-step time grows cubically with n_obs.

### Experiment 3: High-Dimensional Outputs (`experiments/highdim.jl`)

**Goal**: Test with many correlated outputs from a realistic physics-based benchmark.

| Parameter | Value |
|-----------|-------|
| Function | Environmental monitoring model |
| N | 500 |
| d | 4 |
| D | 12 |
| Q | 6 |
| Steps | 100 |
| Seeds | 10 (0-9) |
| n_seed | 15 |
| ℓs | [1.0, 1.5, 2.0, 2.5, 3.0, 3.5] |
| σ2s | [3.0, 3.0, 2.0, 2.0, 1.0, 1.0] |
| R_diag_init | 0.2 |
| beta | 2.0 |
| tune_every | 0 (disabled) |

**Outputs** (`data/highdim/`):
- `comparison.json`, `convergence.png`, `timing.png`
- `metrics.json`: DVC metrics

**Expected result**: With D=12, the kernel-matrix baseline operates on (12*n_obs) sized matrices, making the cubic scaling even more prohibitive. SS-GP scales as O(N*(2*6)^3) = O(N*1728) per step, which remains manageable. Additionally, the Wishart prior on Lambda is now 12x12, allowing the model to learn richer noise structure.

### Comparison Framework

The `run_comparison` function (`src/comparison.jl`) handles the multi-seed comparison loop:

1. For each seed:
   a. Run `setup_experiment` + `run_bo!` (state-space GP with online noise learning)
   b. Run `setup_experiment` again (fresh state, same seed) + `setup_baseline` + `run_bo_baseline!` (kernel-matrix GP with fixed noise)
2. Collect per-seed results: best_value_history, step_times, best_value, n_iterations, total_time
3. Save `comparison.json` (all per-seed data)
4. Generate convergence plot (mean +/- std across seeds) and timing plot (median per-step)

Both methods start from the **same initial observations** (same seed -> same random points, same chain ordering, same seed indices), with the **same structural hyperparameters** (W, ℓs, σ2s). The SS-GP has a Wishart prior on noise precision initialized at E[R] = R_diag_init * I(D); the baseline has R fixed at R_diag_init * I(D).

## Data Pipeline

For both methods:

1. Generate N random points in R^d (standard normal, seeded with `cfg.seed`)
2. Standardize inputs (zero mean, unit variance per dimension)
3. Compute NN chain ordering and inter-point distances Delta
4. Normalize Delta by median (`dist_norm`)
5. Initialize mixing matrix W ~ N(0, 0.25) of size D x Q (seeded with `cfg.seed`)
6. Randomly select n_seed points (sampled with replacement via `rand(rng, 1:N, n_seed)`)
7. Evaluate seed observations, standardize outputs (mean/std from seed observations)

For SS-GP:
8. Build state-space blocks from Delta (A, Q, P, H)
9. Initialize Wishart prior: Lambda ~ Wishart(D+2, I/(nu_0 * R_diag_init))
10. Per BO step: VMP inference (10 iterations) -> update Wishart posterior -> UCB acquisition -> evaluate next point

For KM-GP:
8. Set R = R_diag_init * I(D) (fixed, never updated)
9. Per BO step: build kernel matrix -> Cholesky -> predict -> UCB acquisition -> evaluate next point

## Key Design Decisions

- **Online noise learning in SS-GP** -- Lambda ~ Wishart prior, updated sequentially from VMP posteriors. This allows the noise model to adapt as data accumulates and can learn a full (non-diagonal) precision matrix.
- **Fixed noise in baseline** -- R = R_diag_init * I(D) is constant. This is the standard approach for kernel-matrix GPs where noise is typically a fixed hyperparameter.
- **State-space GP instead of kernel-matrix GP** -- O(N) vs O(N^3) scaling per BO step
- **NN chain ordering** -- Greedy heuristic enabling state-space GP on high-dim inputs; always starts from index 1
- **Additive latent structure** -- Q independent latent GPs mixed by W (LMC)
- **RxInfer for inference** -- Automated message passing on factor graphs, VMP for conjugate Wishart-Normal updates
- **UCB with scalarization** -- Multi-output -> scalar via preference vector s; assumes independent output dims in uncertainty computation
- **No globals** -- All mutable state in BOState / BaselineState structs
- **Hyperparameter tuning disabled** -- ℓs, σ2s, W are fixed throughout all experiments (tune_every=0 in all configs)
- **DVC pipeline** -- Reproducible experiments with tracked parameters and outputs

## File Structure

```
Project.toml              -- Julia package definition + dependencies
params.yaml               -- Experiment hyperparameters (DVC-tracked)
dvc.yaml                  -- DVC pipeline (3 stages: convergence, scaling, highdim)
experiments/
  convergence.jl          -- Experiment 1: convergence comparison (d=6, D=4, N=300)
  scaling.jl              -- Experiment 2: computational scaling (d=6, D=4, N=1000)
  highdim.jl              -- Experiment 3: high-dim outputs (d=4, D=12, N=500)
src/
  RxBayesOpt.jl           -- Module definition, includes, and exports
  types.jl                -- AbstractBOState, ExperimentConfig, BOState, BOResult
  utils.jl                -- row, sqdist, nn_chain_order, blockdiag
  statespace.jl           -- matern32_blocks_from_Δ, additive_multioutput_blocks_from_Δ
  model.jl                -- @model additive_gp_vv (RxInfer model, Wishart noise prior)
  acquisition.jl          -- ucb_acquisition, select_next_point
  hyperparameters.jl      -- log_marginal_likelihood, tune_hyperparameters (disabled in experiments)
  visualization.jl        -- save_plot, plot_bo_step
  bo.jl                   -- setup_experiment, run_bo!, _current_best, save_results
  baseline.jl             -- BaselineState, LMC kernel functions, run_bo_baseline!
  comparison.jl           -- run_comparison, _plot_comparison
  benchmarks.jl           -- hartmann6, make_mo_hartmann, make_environmental
```

## Implementation Details

### State Types

```julia
# Immutable experiment configuration
@kwdef struct ExperimentConfig
    N=100, d=20, Q=8, D=10
    ℓs=[1.0, 1.4, 1.9, 2.5, 3.2, 4.0, 5.0, 6.2]
    σ2s=[5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0]
    β=2.0, s=fill(0.1, 10)
    n_seed=4, steps=200, tune_every=0
    R_diag_init=0.2
    animate=true, log_every=10, seed=0
end

# Mutable state for SS-GP BO loop
mutable struct BOState <: AbstractBOState
    blocks::NamedTuple{(:A, :Q, :P, :H)}  # state-space matrices
    W::Matrix{Float64}                      # mixing matrix D x Q
    R_prior::Any                            # Wishart distribution over precision Lambda = R^{-1}
                                            # Updated each step from VMP posterior
    Y::Vector{Union{Missing, Vector{Float64}}}
    μy::Vector{Float64}                     # output standardization mean
    σy::Vector{Float64}                     # output standardization std
end

# Mutable state for KM-GP baseline
mutable struct BaselineState <: AbstractBOState
    W::Matrix{Float64}                      # mixing matrix D x Q
    ℓs::Vector{Float64}                     # kernel length-scales
    σ2s::Vector{Float64}                    # kernel signal variances
    R::Matrix{Float64}                      # FIXED noise covariance = R_diag_init * I(D)
    dist_norm::Float64                      # median chain distance (for normalizing pairwise distances)
    Xo::Matrix{Float64}                     # candidate points (chain-ordered)
    Y::Vector{Union{Missing, Vector{Float64}}}
    μy::Vector{Float64}
    σy::Vector{Float64}
end

# Result struct
struct BOResult
    best_index, best_value, best_y, observed_indices, n_iterations
    R_learned::Matrix{Float64}              # SS-GP: inv(mean(Wishart posterior)); KM-GP: fixed R
    best_value_history, n_observed_history
    R_diag_history::Vector{Vector{Float64}} # SS-GP: evolving diag; KM-GP: constant diag
    step_times
    method::String                          # "state-space" or "kernel-matrix"
end
```

### RxInfer Model

```julia
@model function additive_gp_vv(Y, P, A, Q, H, prior_Λ)
    Λ ~ prior_Λ                                        # Wishart prior on precision
    fprev ~ MvNormal(μ=zeros(size(P, 1)), Σ=P)
    for i in eachindex(Y)
        f[i] ~ MvNormal(μ=A[i] * fprev, Σ=Q[i])
        my[i] := H * f[i]                              # deterministic observation mean
        Y[i] ~ MvNormalMeanPrecision(my[i], Λ)         # parameterized by PRECISION, not covariance
        fprev = f[i]
    end
end
```

Key points:
- `prior_Λ` is a `Wishart` distribution object, passed as an argument (not data)
- `Λ` is a random variable, making this a **hierarchical** model requiring VMP
- `MvNormalMeanPrecision` parameterizes the likelihood by the precision matrix Lambda (not covariance R)
- The variational constraint `q(Λ, my) = q(Λ)q(my)` factorizes the precision from the means
- `my[i] := ...` uses RxInfer's deterministic node syntax

### Visualization

Plots are saved as both PNG (via GR backend) and TikZ (via PGFPlotsX backend) for direct LaTeX inclusion. The `save_plot` function handles backend switching with `invokelatest` for Julia 1.12 world-age compatibility.

## Performance Metrics

Per-seed results saved to `comparison.json`:
- `best_value_history`: best scalarized value s^T*y after each step (convergence curve)
- `step_times`: wall-clock seconds per BO step (timing curve)
- `best_value`: final best scalarized value
- `n_iterations`: number of BO steps completed
- `total_time`: sum of step_times

DVC metrics saved to `metrics.json`:
- Mean final best values (SS and KM)
- Mean total times (SS and KM)
- Scaling-specific: median step times, KM/SS time ratio

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| N | Number of candidate points (fixed design set) |
| n_obs | Number of observed points (grows during BO) |
| d | Input dimensionality |
| D | Number of output dimensions |
| Q | Number of independent latent GPs |
| ell_q | Length-scale of latent GP q |
| sigma^2_q | Signal variance of latent GP q |
| W | Mixing matrix, R^{D x Q} |
| Lambda | Observation precision matrix = R^{-1}, R^{D x D} (random variable in SS-GP, implicit fixed in KM-GP) |
| R | Observation noise covariance = Lambda^{-1}, R^{D x D} |
| R_diag_init | Initial diagonal value for noise covariance (default: 0.2) |
| nu_0 | Wishart degrees of freedom = D + 2 |
| S_0 | Wishart scale matrix = I_D / (nu_0 * R_diag_init) |
| A_i | State transition matrix at chain position i, R^{2Q x 2Q} |
| Q_i | Process noise covariance at chain position i, R^{2Q x 2Q} |
| P | Stationary (initial) state covariance, R^{2Q x 2Q} |
| H | Observation matrix, R^{D x 2Q} |
| Delta_i | Inter-point distance between chain positions i-1 and i (normalized by median) |
| dist_norm | Median of raw chain distances, used to normalize both Delta and pairwise distances |
| x_i | Latent state vector at chain position i, R^{2Q} |
| f_i^{(q)} | Value of latent GP q at chain position i |
| m_i | Observation mean = H * x_i, R^D |
| s | Scalarization preference vector, R^D (default: uniform 1/D) |
| beta | UCB exploration weight (default: 2.0) |

## Key References to Cite

- Hartikainen & Sarkka (2010) -- Kalman filtering and smoothing solutions to temporal GP regression
- Sarkka & Solin (2019) -- Applied Stochastic Differential Equations (state-space GPs)
- Alvarez et al. (2012) -- Kernels for vector-valued functions (LMC)
- Bagaev & de Vries (2023) -- RxInfer: A Julia package for reactive real-time Bayesian inference
- Snoek et al. (2012) -- Practical Bayesian optimization of machine learning algorithms
- Srinivas et al. (2010) -- Gaussian process optimization in the bandit setting (GP-UCB)
- Dixon & Szego (1978) -- The Global Optimization Problem (Hartmann-6 benchmark)
- Bliznyuk et al. (2008) -- Bayesian calibration of mechanistic aquatic biogeochemical models (environmental benchmark)
- Maddox et al. (2021) -- Bayesian Optimization with High-Dimensional Outputs, NeurIPS (environmental benchmark, multi-output BO)
