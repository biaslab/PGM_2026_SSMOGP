"""
    BaselineState <: AbstractBOState

Mutable state for the LMC kernel-matrix GP baseline.

Uses the same LMC model as the state-space method (same W, ℓs, σ2s)
but performs inference via exact kernel matrix + Cholesky decomposition.

# Fields
- `W::Matrix{Float64}`: mixing matrix (D × Q), same as state-space method
- `ℓs::Vector{Float64}`: length-scales per latent GP
- `σ2s::Vector{Float64}`: signal variances per latent GP
- `R::Matrix{Float64}`: fixed D×D observation noise covariance
- `dist_norm::Float64`: distance normalization factor (median of chain distances)
- `Xo::Matrix{Float64}`: ordered input matrix (N × d)
- `Y::Vector{Union{Missing, Vector{Float64}}}`: observations (standardized)
- `μy::Vector{Float64}`: mean used for standardization
- `σy::Vector{Float64}`: std used for standardization
"""
mutable struct BaselineState <: AbstractBOState
    W::Matrix{Float64}
    ℓs::Vector{Float64}
    σ2s::Vector{Float64}
    R::Matrix{Float64}
    dist_norm::Float64
    Xo::Matrix{Float64}
    Y::Vector{Union{Missing, Vector{Float64}}}
    μy::Vector{Float64}
    σy::Vector{Float64}
end

"""
    _matern32_kernel(r, ℓ, σ2)

Scalar Matérn 3/2 kernel: σ2 * (1 + √3 r/ℓ) * exp(-√3 r/ℓ).
"""
@inline function _matern32_kernel(r::Float64, ℓ::Float64, σ2::Float64)
    z = sqrt(3.0) * r / ℓ
    σ2 * (1.0 + z) * exp(-z)
end

"""
    _pairwise_distances(X1, X2)

Compute Euclidean distance matrix (n1 × n2) between rows of X1 and X2.
"""
function _pairwise_distances(X1::AbstractMatrix, X2::AbstractMatrix)
    n1 = size(X1, 1)
    n2 = size(X2, 1)
    D = zeros(n1, n2)
    for j in 1:n2, i in 1:n1
        D[i, j] = sqrt(sqdist(row(X1, i), row(X2, j)))
    end
    D
end

"""
    _lmc_kernel_matrix(X, W, ℓs, σ2s, R, dist_norm)

Build the full (D*n × D*n) LMC covariance matrix for observed points X.

K = Σ_q kron(W[:,q]*W[:,q]', K_q) + kron(R, I(n))

where K_q[i,j] = matern32(dist/dist_norm, ℓs[q], σ2s[q]).
Distances are normalized by dist_norm so that ℓs have the same meaning
as in the state-space representation.
"""
function _lmc_kernel_matrix(X::AbstractMatrix, W::Matrix{Float64},
                            ℓs::Vector{Float64}, σ2s::Vector{Float64},
                            R::Matrix{Float64}, dist_norm::Float64)
    n = size(X, 1)
    D = size(W, 1)
    Q = size(W, 2)

    # Pairwise distances, normalized
    dists = _pairwise_distances(X, X) ./ dist_norm

    # Build full covariance
    K = zeros(D * n, D * n)

    # Add latent kernel contributions: Σ_q kron(W[:,q]*W[:,q]', K_q)
    for q in 1:Q
        wq = W[:, q]
        Bq = wq * wq'  # D × D
        for j in 1:n, i in 1:n
            kval = _matern32_kernel(dists[i, j], ℓs[q], σ2s[q])
            for b in 1:D, a in 1:D
                K[(a-1)*n + i, (b-1)*n + j] += Bq[a, b] * kval
            end
        end
    end

    # Add noise: kron(R, I(n))
    for j in 1:D, i in 1:D
        rij = R[i, j]
        if rij != 0.0
            for k in 1:n
                K[(i-1)*n + k, (j-1)*n + k] += rij
            end
        end
    end

    # Ensure symmetry
    Symmetric(K)
end

"""
    _lmc_cross_kernel(X_test, X_obs, W, ℓs, σ2s, dist_norm)

Build the cross-covariance (D*n_test × D*n_obs) between test and observed points.
No noise added to cross-covariance.
"""
function _lmc_cross_kernel(X_test::AbstractMatrix, X_obs::AbstractMatrix,
                           W::Matrix{Float64}, ℓs::Vector{Float64},
                           σ2s::Vector{Float64}, dist_norm::Float64)
    n_test = size(X_test, 1)
    n_obs = size(X_obs, 1)
    D = size(W, 1)
    Q = size(W, 2)

    dists = _pairwise_distances(X_test, X_obs) ./ dist_norm

    Kx = zeros(D * n_test, D * n_obs)
    for q in 1:Q
        wq = W[:, q]
        Bq = wq * wq'
        for j in 1:n_obs, i in 1:n_test
            kval = _matern32_kernel(dists[i, j], ℓs[q], σ2s[q])
            for b in 1:D, a in 1:D
                Kx[(a-1)*n_test + i, (b-1)*n_obs + j] += Bq[a, b] * kval
            end
        end
    end
    Kx
end

"""
    _lmc_self_variance(x, W, ℓs, σ2s)

Compute the prior variance at a single point (D-vector of marginal variances).
At distance 0, matern32(0, ℓ, σ2) = σ2, so:
    var_d = Σ_q W[d,q]^2 * σ2s[q]
"""
function _lmc_self_variance(W::Matrix{Float64}, σ2s::Vector{Float64})
    D, Q = size(W)
    v = zeros(D)
    for q in 1:Q
        for d in 1:D
            v[d] += W[d, q]^2 * σ2s[q]
        end
    end
    v
end

"""
    _lmc_predict(bl_state::BaselineState)

Predict at all N candidate points using LMC kernel-matrix GP.

Returns:
- `μ_pred`: Vector of D-vectors, predictive means (standardized scale)
- `σ_pred`: Vector of D-vectors, predictive std devs (standardized scale)
"""
function _lmc_predict(bl_state::BaselineState)
    N = size(bl_state.Xo, 1)
    D = size(bl_state.W, 1)
    observed = findall(!ismissing, bl_state.Y)
    n_obs = length(observed)

    X_obs = bl_state.Xo[observed, :]

    # Stack observed outputs into (D*n_obs) vector
    # Layout: [y1_d1, y2_d1, ..., yn_d1, y1_d2, ..., yn_dD]
    y_obs = zeros(D * n_obs)
    for (idx, k) in enumerate(observed)
        for d in 1:D
            y_obs[(d-1)*n_obs + idx] = bl_state.Y[k][d]
        end
    end

    # Build and factorize observation kernel matrix
    K_obs = _lmc_kernel_matrix(X_obs, bl_state.W, bl_state.ℓs, bl_state.σ2s,
                               bl_state.R, bl_state.dist_norm)
    # Add jitter for numerical stability
    K_obs_jit = Matrix(K_obs) + 1e-8 * I(D * n_obs)
    L = cholesky(Hermitian(K_obs_jit))
    α = L \ y_obs  # K_obs^{-1} y_obs

    # Prior self-variance (same for all points)
    prior_var = _lmc_self_variance(bl_state.W, bl_state.σ2s)

    # Predict at each candidate point
    μ_pred = Vector{Vector{Float64}}(undef, N)
    σ_pred = Vector{Vector{Float64}}(undef, N)

    for i in 1:N
        X_star = bl_state.Xo[i:i, :]  # 1 × d
        # Cross-covariance: (D*1 × D*n_obs)
        Kx = _lmc_cross_kernel(X_star, X_obs, bl_state.W, bl_state.ℓs,
                               bl_state.σ2s, bl_state.dist_norm)

        # Predictive mean: Kx * K_obs^{-1} * y_obs
        μ_i = Kx * α  # D-vector (one entry per output)

        # Predictive variance: diag(K_** - Kx * K_obs^{-1} * Kx')
        v = L.L \ Kx'  # (D*n_obs × D) — solve triangular
        var_i = prior_var .- vec(sum(v .^ 2, dims=1))
        var_i .= max.(var_i, 1e-10)

        # Extract per-output mean and std
        μ_d = Float64[μ_i[d] for d in 1:D]
        σ_d = Float64[sqrt(var_i[d]) for d in 1:D]

        μ_pred[i] = μ_d
        σ_pred[i] = σ_d
    end

    (; μ_pred, σ_pred)
end

"""
    setup_baseline(cfg::ExperimentConfig, setup_data) -> BaselineState

Create a BaselineState from the output of `setup_experiment`.
Copies W, observations, and standardization parameters.
Uses the same ℓs, σ2s from cfg and fixed R = R_diag_init * I(D).
"""
function setup_baseline(cfg::ExperimentConfig, setup_data)
    W_copy = copy(setup_data.W)
    Y_copy = copy(setup_data.Y)
    μy_copy = copy(setup_data.μy)
    σy_copy = copy(setup_data.σy)
    R = cfg.R_diag_init * Matrix{Float64}(I(cfg.D))
    BaselineState(W_copy, copy(cfg.ℓs), copy(cfg.σ2s), R,
                  setup_data.dist_norm, copy(setup_data.Xo),
                  Y_copy, μy_copy, σy_copy)
end

"""
    baseline_ucb_acquisition(bl_state::BaselineState, cfg::ExperimentConfig)

Compute UCB acquisition using LMC kernel-matrix GP predictions.
Returns the same named tuple format as `ucb_acquisition`.
"""
function baseline_ucb_acquisition(bl_state::BaselineState, cfg::ExperimentConfig)
    N = size(bl_state.Xo, 1)
    D = cfg.D
    s = cfg.s

    pred = _lmc_predict(bl_state)

    # Convert to original scale
    μ_pred = [pred.μ_pred[i] .* bl_state.σy .+ bl_state.μy for i in 1:N]
    σ_pred = [pred.σ_pred[i] .* bl_state.σy for i in 1:N]

    # Scalarized UCB
    μs = [dot(s, μ_pred[i]) for i in 1:N]
    σs = [sqrt(sum((s[j] * σ_pred[i][j])^2 for j in 1:D)) for i in 1:N]
    ucb = μs .+ cfg.β .* σs

    (; ucb, μ_pred, σ_pred, μs, σs)
end

"""
    run_bo_baseline!(cfg::ExperimentConfig, eval_fn; bl_state, Ytrue) -> (; result, frames)

Run the BO loop using the LMC kernel-matrix GP as the surrogate.

Same loop structure as `run_bo!`:
1. Predict using LMC kernel-matrix GP (Cholesky)
2. Compute UCB acquisition
3. Select next point, evaluate, update observations
"""
function run_bo_baseline!(cfg::ExperimentConfig, eval_fn; bl_state::BaselineState, Ytrue)
    N = cfg.N
    Xo = bl_state.Xo
    frames = cfg.animate ? [] : nothing

    best_value_history = Float64[]
    n_observed_history = Int[]
    R_diag_history = Vector{Float64}[]
    step_times = Float64[]

    n_completed = 0
    done = false
    for step in 1:cfg.steps
        t_step = @elapsed begin
            local acq
            try
                acq = baseline_ucb_acquisition(bl_state, cfg)
            catch e
                @warn "LMC prediction failed at step $step, stopping early" exception=e
                done = true
                @goto next_step
            end

            k = select_next_point(acq.ucb, bl_state.Y)
            if k == 0
                done = true
            else
                if cfg.animate
                    plt = plot_bo_step(step, k, bl_state, acq, Ytrue, cfg)
                    push!(frames, plt)
                end

                y_new = eval_fn(row(Xo, k))
                bl_state.Y[k] = (y_new .- bl_state.μy) ./ bl_state.σy

                n_completed = step

                # Track per-step metrics
                best = _current_best(bl_state, cfg)
                n_obs = length(findall(!ismissing, bl_state.Y))

                push!(best_value_history, best.value)
                push!(n_observed_history, n_obs)
                push!(R_diag_history, diag(bl_state.R))

                if step % cfg.log_every == 0
                    @info "Baseline Step $step/$(cfg.steps)" n_observed=n_obs best_value=round(best.value; digits=4)
                end
            end
            @label next_step
        end # @elapsed
        push!(step_times, t_step)
        done && break
    end

    best = _current_best(bl_state, cfg)
    observed = findall(!ismissing, bl_state.Y)
    R_learned = copy(bl_state.R)

    result = BOResult(best.index, best.value, best.y, observed, n_completed, R_learned,
                      best_value_history, n_observed_history, R_diag_history,
                      step_times, "kernel-matrix")
    (; result, frames)
end
