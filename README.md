# RxBayesOpt

**Scalable Multi-Output Bayesian Optimization via State-Space GPs and Reactive Message Passing**

Multi-output Bayesian Optimization using state-space Gaussian Processes and reactive message passing inference via [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl).

## Key Ideas

1. **O(N) surrogate model** — Standard GP-based BO uses kernel matrices with O(N³) cost. We represent the Matérn 3/2 GP as a state-space model (linear dynamical system), reducing inference to a single Kalman filter pass in O(N).

2. **Multi-output via state-space LMC** — Multiple correlated outputs are modeled through Q independent latent state-space GPs combined by a mixing matrix W (Linear Model of Coregionalization). The block-diagonal structure preserves the O(N) scaling.

3. **NN chain ordering for high-dimensional inputs** — State-space GPs require 1D-ordered inputs. A greedy nearest-neighbor chain maps high-dimensional points onto a 1D sequence that preserves local distance structure, with inter-point distances serving as the state-space time steps.

4. **Unified inference via message passing** — The entire model (latent GP states, predictions, per-output noise) is expressed as a single probabilistic model in RxInfer. Inference is automated message passing on the factor graph — no hand-coded Kalman filter needed.

5. **Partial observations via FFG modularity** — Per-output scalar noise factors allow natural handling of missing/partial observations. When an output is unobserved, the corresponding factor simply passes through a vague message — no special-case code required. This is a direct consequence of the factor graph structure.

6. **Reactive architecture** — RxInfer's reactive message passing framework enables potential streaming/online BO where the surrogate updates incrementally as new observations arrive, rather than re-running inference from scratch.

## Why This Matters

| Aspect | Standard GP-BO | This work |
|--------|---------------|-----------|
| Surrogate cost | O(N³) kernel matrix | O(N) Kalman filtering |
| Multi-output | Separate GPs or kernel-matrix LMC | State-space LMC (block-diagonal) |
| Partial observations | Requires variable-size covariance matrices | Natural via FFG message passing |
| Noise model | Fixed or point-estimated | Per-output scalar precision |
| Inference | Custom GP algebra | Declarative model + automated message passing |
| High-dim inputs | Native (but O(N³)) | NN chain ordering heuristic |
| Online updates | Full recomputation | Incremental (reactive message passing) |

## Methods

### State-Space Gaussian Processes

A GP with Matérn 3/2 kernel can be exactly represented as a **linear dynamical system** (LDS / state-space model) when inputs are one-dimensional. The continuous-time SDE is discretized into transition matrices **A**, process noise covariances **Q**, and an observation vector **H**, yielding a Kalman-filter-equivalent formulation. This reduces GP inference from O(N³) to O(N).

### Nearest-Neighbor Chain Ordering

State-space GPs require a 1D ordering of data points. For high-dimensional inputs, a **greedy nearest-neighbor chain** is used: starting from an arbitrary point, the next point in the chain is always the closest unvisited neighbor (in Euclidean distance). The inter-point distances Δ become the "time steps" of the state-space model. This is a heuristic that maps multi-dimensional inputs onto a 1D chain while preserving local structure.

### Multi-Output Additive Structure (Linear Model of Coregionalization)

Multiple latent GP components (Q=6 in the default experiment) are combined through a **mixing matrix W** (D×Q) to produce D output dimensions. Each latent GP has its own length-scale ℓ and variance σ² parameters. The block-diagonal structure keeps the state-space representation tractable: the full state is the concatenation of all latent states, and the observation model is `y = H_big * f` where `H_big = W ⊗ h'` encodes the mixing.

### RxInfer Message Passing Inference

The state-space GP is expressed as an `@model` in RxInfer, which performs message passing on the corresponding factor graph. Rather than hand-coding a Kalman filter, the model is declared probabilistically and inference is automated. Per-output scalar noise factors (`NormalMeanPrecision`) allow partial observations to be handled naturally — missing entries simply receive vague messages.

### Partial Observations

Outputs can be partially observed (e.g., sensor groups that each measure a subset of output dimensions). The per-output noise factorization in the factor graph means that unobserved entries are handled automatically by message passing — no variable-size covariance matrices or special indexing needed. This is a key advantage of the FFG formulation over kernel-matrix approaches.

### UCB Acquisition with Scalarization

The multi-output predictions are scalarized via a preference vector **s** (dot product with the predicted mean and variance), and then a standard **Upper Confidence Bound** (UCB) acquisition function selects the next point to evaluate:

```
UCB(x) = sᵀμ(x) + β · √(sᵀdiag(Σ(x))s)
```

### Hyperparameter Tuning (Type-II ML)

An LBFGS-based optimizer maximizes the log marginal likelihood (computed via a Kalman filter forward pass) with respect to kernel length-scales, signal variances, the mixing matrix W, and observation noise. Controlled by `ExperimentConfig.tune_every` (0 = disabled).

## Experiment: Partial Observations on Environmental Monitoring

The main experiment (`experiments/partial_obs.jl`) runs a **4-way comparison** on an environmental monitoring benchmark (Bliznyuk et al. 2008):

- **SS-GP Full** — State-space GP, all outputs observed
- **SS-GP Partial** — State-space GP, sensor-group partial observations
- **KM-GP Full** — Kernel-matrix GP baseline, all outputs observed
- **KM-GP Partial** — Kernel-matrix GP baseline, sensor-group partial observations

Default configuration: d=4 inputs, D=12 outputs (3 spatial × 4 temporal), Q=6 latent GPs, N=500 candidates, 150 BO steps, 5 seeds.

Outputs in `data/partial_obs/`:
- `comparison.json` — Per-seed results (best value history, step times) for all 4 methods
- `metrics.json` — Aggregated metrics across seeds
- `convergence.png` — Best value over BO steps (mean ± std across seeds)
- `timing.png` — Median time per step (log scale)

## Project Structure

```
Project.toml                — Julia package definition + dependencies
params.yaml                 — Experiment hyperparameters (DVC-tracked)
dvc.yaml                    — DVC pipeline definition
experiments/
  partial_obs.jl            — 4-way comparison: SS/KM × full/partial obs
  profile.jl                — Profiling script
src/
  RxBayesOpt.jl             — Module definition, includes, and exports
  types.jl                  — ExperimentConfig, POState, BaselinePOState, BOResult
  utils.jl                  — row, sqdist, nn_chain_order, blockdiag
  statespace.jl             — Matérn 3/2 state-space discretization + LMC blocks
  model.jl                  — RxInfer @model (additive_gp_po)
  acquisition.jl            — UCB acquisition + point selection
  hyperparameters.jl        — Log marginal likelihood + LBFGS tuning
  visualization.jl          — 3-panel BO step visualization
  bo.jl                     — setup_experiment, save_results
  baseline.jl               — LMC kernel-matrix GP (O(N³) baseline)
  partial_obs.jl            — Partial-obs SS-GP + KM-GP + comparison runner
  benchmarks.jl             — Hartmann-6, multi-output Hartmann, environmental model
```

## Dependencies

- **RxInfer.jl** — Reactive message passing inference
- **Distributions.jl** — Probability distributions
- **LinearAlgebra** — Matrix operations
- **Statistics** — Mean, std, median
- **Optim.jl** — Hyperparameter optimization (LBFGS)
- **Plots.jl** / **PGFPlotsX.jl** — Visualization (PNG + TikZ)
- **JSON** / **YAML** — Metrics export and config loading
- **PDMats** / **MatrixCorrectionTools** — PD matrix utilities
- **Random** — Reproducible RNG

## Running

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run via DVC pipeline
dvc repro

# Or run the experiment directly
julia --project=. experiments/partial_obs.jl
```
