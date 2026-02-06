"""
    ExperimentConfig

Immutable configuration for a multi-output Bayesian optimization experiment.

# Fields
- `N::Int`: number of candidate points
- `d::Int`: input dimensionality
- `Q::Int`: number of latent GPs
- `D::Int`: number of outputs
- `ℓs::Vector{Float64}`: length-scales for each latent GP
- `σ2s::Vector{Float64}`: signal variances for each latent GP
- `β::Float64`: UCB exploration weight
- `s::Vector{Float64}`: scalarization vector (length D)
- `n_seed::Int`: number of seed observations
- `steps::Int`: number of BO iterations
- `tune_every::Int`: re-tune hyperparameters every N steps (0 = disabled)
- `R_diag_init::Float64`: initial observation noise variance
- `animate::Bool`: whether to produce an animation
- `log_every::Int`: log progress every N steps
- `seed::Int`: random seed
"""
@kwdef struct ExperimentConfig
    N::Int            = 100
    d::Int            = 20
    Q::Int            = 8
    D::Int            = 10
    ℓs::Vector{Float64}  = [1.0, 1.4, 1.9, 2.5, 3.2, 4.0, 5.0, 6.2]
    σ2s::Vector{Float64} = [5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0]
    β::Float64        = 2.0
    s::Vector{Float64}   = fill(0.1, 10)
    n_seed::Int       = 4
    steps::Int        = 200
    tune_every::Int   = 0
    R_diag_init::Float64 = 0.2
    animate::Bool     = true
    log_every::Int    = 10
    seed::Int         = 0
end

"""
    BOState

Mutable state that evolves during the BO loop.

# Fields
- `blocks`: named tuple `(A, Q, P, H)` from the state-space GP construction
- `W::Matrix{Float64}`: mixing matrix (D × Q)
- `R::Diagonal{Float64}`: observation noise covariance
- `Y::Vector{Union{Missing, Vector{Float64}}}`: observations (standardized)
- `μy::Vector{Float64}`: mean used for standardization
- `σy::Vector{Float64}`: std used for standardization
"""
mutable struct BOState
    blocks::NamedTuple{(:A, :Q, :P, :H)}
    W::Matrix{Float64}
    R::Diagonal{Float64, Vector{Float64}}
    Y::Vector{Union{Missing, Vector{Float64}}}
    μy::Vector{Float64}
    σy::Vector{Float64}
end

"""
    BOResult

Summary of a completed Bayesian optimization run.

# Fields
- `best_index::Int`: chain index of the best observed point (by scalarized value)
- `best_value::Float64`: scalarized value at the best point
- `best_y::Vector{Float64}`: output vector at the best point (original scale)
- `observed_indices::Vector{Int}`: all observed chain indices
- `n_iterations::Int`: number of BO iterations completed
"""
struct BOResult
    best_index::Int
    best_value::Float64
    best_y::Vector{Float64}
    observed_indices::Vector{Int}
    n_iterations::Int
end

"""
    print_config(cfg::ExperimentConfig)

Print experiment configuration to the log.
"""
function print_config(cfg::ExperimentConfig)
    @info "Experiment Configuration" cfg.N cfg.d cfg.Q cfg.D cfg.β cfg.n_seed cfg.steps cfg.tune_every cfg.R_diag_init cfg.animate cfg.log_every cfg.seed
    @info "Length-scales" cfg.ℓs
    @info "Signal variances" cfg.σ2s
end

"""
    print_summary(result::BOResult)

Print a summary of the BO run results.
"""
function print_summary(result::BOResult)
    @info "BO Run Complete" result.n_iterations n_observed=length(result.observed_indices) best_index=result.best_index best_value=round(result.best_value; digits=4)
    @info "Best output vector" result.best_y
end
