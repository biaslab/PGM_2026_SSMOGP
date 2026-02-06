# RxBayesOpt

Multi-output Bayesian Optimization using state-space Gaussian Processes and reactive message passing inference via [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl).

## Overview

This project implements Bayesian Optimization (BO) for expensive black-box functions with **multiple outputs** in **high-dimensional input spaces**. Instead of the standard kernel-matrix GP (which scales O(N³)), the surrogate model is a **state-space GP** that scales linearly O(N) by exploiting the Markov structure of the Matérn 3/2 covariance function.

Inference is performed via **variational message passing** in RxInfer, which opens the door for reactive, online updates as new observations arrive.

## Methods

### State-Space Gaussian Processes

A GP with Matérn 3/2 kernel can be exactly represented as a **linear dynamical system** (LDS / state-space model) when inputs are one-dimensional. The continuous-time SDE is discretized into transition matrices **A**, process noise covariances **Q**, and an observation vector **H**, yielding a Kalman-filter-equivalent formulation. This reduces GP inference from O(N³) to O(N).

### Nearest-Neighbor Chain Ordering

State-space GPs require a 1D ordering of data points. For high-dimensional inputs (d=20 in the experiment), a **greedy nearest-neighbor chain** is used: starting from an arbitrary point, the next point in the chain is always the closest unvisited neighbor (in Euclidean distance). The inter-point distances Δ become the "time steps" of the state-space model. This is a heuristic that maps multi-dimensional inputs onto a 1D chain while preserving local structure.

### Multi-Output Additive Structure (Linear Model of Coregionalization)

Multiple latent GP components (Q=8) are combined through a **mixing matrix W** (D×Q) to produce D=10 output dimensions. Each latent GP has its own length-scale ℓ and variance σ² parameters. The block-diagonal structure keeps the state-space representation tractable: the full state is the concatenation of all latent states, and the observation model is `y = H_big * f` where `H_big = W ⊗ h'` encodes the mixing.

### RxInfer Message Passing Inference

The state-space GP is expressed as an `@model` in RxInfer, which performs (variational) message passing on the corresponding factor graph. This is the key novelty: rather than hand-coding a Kalman filter, the model is declared probabilistically and inference is automated. RxInfer's reactive architecture also enables potential streaming/online BO updates without re-running inference from scratch.

### UCB Acquisition with Scalarization

The multi-output predictions are scalarized via a preference vector **s** (dot product with the predicted mean and variance), and then a standard **Upper Confidence Bound** (UCB) acquisition function selects the next point to evaluate:

```
UCB(x) = sᵀμ(x) + β · √(sᵀdiag(Σ(x))s)
```

### Hyperparameter Tuning (Type-II ML)

An LBFGS-based optimizer maximizes the log marginal likelihood (computed via a Kalman filter forward pass) with respect to kernel length-scales, signal variances, the mixing matrix W, and observation noise. Controlled by `ExperimentConfig.tune_every` (0 = disabled).

## What is Novel

1. **Multi-output BO via state-space GP + message passing** — Combining the O(N) state-space GP formulation with RxInfer's automated message passing for Bayesian optimization is, to our knowledge, a new approach.
2. **Nearest-neighbor chain ordering** — A practical heuristic enabling state-space GPs in high-dimensional input spaces without dimensionality reduction.
3. **Reactive inference for BO** — Using RxInfer opens the path toward truly online/streaming Bayesian optimization where the surrogate updates incrementally.

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
- **Random** — Reproducible RNG

## Running

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. experiments.jl
```

This produces:
- Config printout at start
- Progress logs every 10 steps
- Results summary at end
- `bo_multioutput_chain.gif` — animation of the BO loop showing predictions, UCB, and per-output fits over 200 steps
