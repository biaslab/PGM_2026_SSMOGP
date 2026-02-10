"""
    AbstractBOState

Abstract supertype for Bayesian optimization state objects.
Subtypes must have fields `Y`, `μy`, `σy`.
"""
abstract type AbstractBOState end

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
    obs_pattern::Symbol  = :full
    obs_frac::Float64    = 1.0
end

"""
    POState <: AbstractBOState

Mutable state for partial-observation BO with per-output scalar noise.

# Fields
- `blocks`: named tuple `(A, Q, P, H)` from the state-space GP construction
- `W::Matrix{Float64}`: mixing matrix (D × Q)
- `τ::Vector{Float64}`: per-output noise precisions (length D)
- `e_vecs::Vector{Vector{Float64}}`: standard basis vectors [e_1, ..., e_D]
- `mask::BitMatrix`: N × D observation mask (true = observed)
- `Y::Vector{Union{Missing, Vector{Float64}}}`: point-level observations (for BO logic)
- `Y_flat::Vector{Union{Missing, Float64}}`: flat N*D vector (for model)
- `μy::Vector{Float64}`: mean used for standardization
- `σy::Vector{Float64}`: std used for standardization
"""
mutable struct POState <: AbstractBOState
    blocks::NamedTuple{(:A, :Q, :P, :H)}
    W::Matrix{Float64}
    τ::Vector{Float64}
    e_vecs::Vector{Vector{Float64}}
    mask::BitMatrix
    Y::Vector{Union{Missing, Vector{Float64}}}
    Y_flat::Vector{Union{Missing, Float64}}
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
- `R_learned::Matrix{Float64}`: posterior mean of observation noise covariance
- `best_value_history::Vector{Float64}`: best scalarized value after each step
- `n_observed_history::Vector{Int}`: number of observed points after each step
- `R_diag_history::Vector{Vector{Float64}}`: diagonal of R posterior mean after each step
- `step_times::Vector{Float64}`: wall-clock time (seconds) for each BO step
- `method::String`: identifier for the surrogate model method
"""
struct BOResult
    best_index::Int
    best_value::Float64
    best_y::Vector{Float64}
    observed_indices::Vector{Int}
    n_iterations::Int
    R_learned::Matrix{Float64}
    best_value_history::Vector{Float64}
    n_observed_history::Vector{Int}
    R_diag_history::Vector{Vector{Float64}}
    step_times::Vector{Float64}
    method::String
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
    total_time = sum(result.step_times)
    @info "BO Run Complete" method=result.method result.n_iterations n_observed=length(result.observed_indices) best_index=result.best_index best_value=round(result.best_value; digits=4) total_time=round(total_time; digits=2)
    @info "Best output vector" result.best_y
    @info "Learned R (posterior mean diagonal)" round.(diag(result.R_learned); digits=4)
    @info "History" steps_tracked=length(result.best_value_history) final_best=round(result.best_value_history[end]; digits=4)
end
