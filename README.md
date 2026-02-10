# RxBayesOpt

**Scalable Multi-Output Bayesian Optimization via State-Space GPs and Reactive Message Passing**

Multi-output Bayesian Optimization using state-space Gaussian Processes and reactive message passing inference via [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl).

## Key Ideas

1. **O(N) surrogate model** — Standard GP-based BO uses kernel matrices with O(N³) cost. We represent the Matérn 3/2 GP as a state-space model (linear dynamical system), reducing inference to a single Kalman filter pass in O(N).

2. **Multi-output via state-space LMC** — Multiple correlated outputs are modeled through Q independent latent state-space GPs combined by a mixing matrix W (Linear Model of Coregionalization). The block-diagonal structure preserves the O(N) scaling.

3. **NN chain ordering for high-dimensional inputs** — State-space GPs require 1D-ordered inputs. A greedy nearest-neighbor chain maps high-dimensional points (d=20) onto a 1D sequence that preserves local distance structure, with inter-point distances serving as the state-space time steps.

4. **Unified inference via message passing** — The entire model (latent GP states, predictions, noise covariance) is expressed as a single probabilistic model in RxInfer. Inference is automated variational message passing on the factor graph — no hand-coded Kalman filter needed.

5. **Online Bayesian noise learning** — The observation noise covariance R is given an InverseWishart prior and learned jointly with the GP states through message passing. The posterior from each BO step becomes the prior for the next, giving fully Bayesian online adaptation of the full D×D noise covariance at zero extra cost.

6. **Reactive architecture** — RxInfer's reactive message passing framework enables potential streaming/online BO where the surrogate updates incrementally as new observations arrive, rather than re-running inference from scratch.

## Why This Matters

| Aspect | Standard GP-BO | This work |
|--------|---------------|-----------|
| Surrogate cost | O(N³) kernel matrix | O(N) Kalman filtering |
| Multi-output | Separate GPs or kernel-matrix LMC | State-space LMC (block-diagonal) |
| Noise model | Fixed or point-estimated | Full Bayesian (InverseWishart posterior) |
| Inference | Custom GP algebra | Declarative model + automated message passing |
| High-dim inputs | Native (but O(N³)) | NN chain ordering heuristic |
| Online updates | Full recomputation | Incremental (reactive message passing) |

## Methods

### State-Space Gaussian Processes

A GP with Matérn 3/2 kernel can be exactly represented as a **linear dynamical system** (LDS / state-space model) when inputs are one-dimensional. The continuous-time SDE is discretized into transition matrices **A**, process noise covariances **Q**, and an observation vector **H**, yielding a Kalman-filter-equivalent formulation. This reduces GP inference from O(N³) to O(N).

### Nearest-Neighbor Chain Ordering

State-space GPs require a 1D ordering of data points. For high-dimensional inputs (d=20 in the experiment), a **greedy nearest-neighbor chain** is used: starting from an arbitrary point, the next point in the chain is always the closest unvisited neighbor (in Euclidean distance). The inter-point distances Δ become the "time steps" of the state-space model. This is a heuristic that maps multi-dimensional inputs onto a 1D chain while preserving local structure.

### Multi-Output Additive Structure (Linear Model of Coregionalization)

Multiple latent GP components (Q=8) are combined through a **mixing matrix W** (D×Q) to produce D=10 output dimensions. Each latent GP has its own length-scale ℓ and variance σ² parameters. The block-diagonal structure keeps the state-space representation tractable: the full state is the concatenation of all latent states, and the observation model is `y = H_big * f` where `H_big = W ⊗ h'` encodes the mixing.

### RxInfer Message Passing Inference

The state-space GP is expressed as an `@model` in RxInfer, which performs (variational) message passing on the corresponding factor graph. Rather than hand-coding a Kalman filter, the model is declared probabilistically and inference is automated. RxInfer's reactive architecture also enables potential streaming/online BO updates without re-running inference from scratch.

### Online Observation Noise Learning

The observation noise covariance R is treated as a random variable with an **InverseWishart prior**. At each BO step, variational message passing infers a posterior over R jointly with the latent GP states. This posterior becomes the prior for the next step, yielding a fully Bayesian online estimate of the full D×D noise covariance. This is in contrast to standard approaches that either fix R or re-estimate it via point optimization.

### UCB Acquisition with Scalarization

The multi-output predictions are scalarized via a preference vector **s** (dot product with the predicted mean and variance), and then a standard **Upper Confidence Bound** (UCB) acquisition function selects the next point to evaluate:

```
UCB(x) = sᵀμ(x) + β · √(sᵀdiag(Σ(x))s)
```

### Hyperparameter Tuning (Type-II ML)

An LBFGS-based optimizer maximizes the log marginal likelihood (computed via a Kalman filter forward pass) with respect to kernel length-scales, signal variances, the mixing matrix W, and observation noise. Controlled by `ExperimentConfig.tune_every` (0 = disabled).

## Project Structure

```
Project.toml              — Julia package definition + dependencies
experiments.jl            — Entry point: defines eval_blackbox, config, runs BO
src/
  RxBayesOpt.jl           — Module definition, includes, and exports
  types.jl                — ExperimentConfig, BOState, BOResult structs
  utils.jl                — row, sqdist, nn_chain_order, blockdiag
  statespace.jl           — State-space GP discretization (Matérn 3/2 + LMC)
  model.jl                — RxInfer @model definition
  acquisition.jl          — UCB acquisition function
  hyperparameters.jl      — Log marginal likelihood + LBFGS tuning
  visualization.jl        — 3-panel BO step visualization
  bo.jl                   — setup_experiment, run_bo! (main BO loop)
```

## Dependencies

- **RxInfer.jl** — Reactive message passing inference
- **Distributions.jl** — Probability distributions
- **LinearAlgebra** — Matrix operations
- **Statistics** — Mean, std, median
- **Optim.jl** — Hyperparameter optimization (LBFGS)
- **Plots.jl** — Visualization and GIF generation
- **JSON** — Performance metrics export
- **Random** — Reproducible RNG

## Running

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. experiments.jl
```

This produces outputs in `data/`:
- `data/metrics.json` — Per-step convergence history (best value, observations, R diagonal), final results, and experiment config. Ready for paper figures.
- `data/bo_last_state.png` — Snapshot of the final BO step (3-panel: predictions, UCB, per-output fits)
- `data/bo_animation.gif` — Animated GIF of all BO steps
