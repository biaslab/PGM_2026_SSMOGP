# Variational Sparse GP (LMC) baseline — Hensman et al. 2013, generalized to LMC.
#
# A third "scalable LMC GP" baseline alongside SS-LMC (state-space, O(N)) and
# KM-LMC (full kernel matrix, O((D·N)³)). SVGP introduces M ≪ N inducing
# pseudo-points and scales as O(N · M²) per fit. Inducing locations Z are
# shared across the Q latents (a common multi-output simplification) and are
# fixed at a random subset of the training inputs.
#
# For the Gaussian-likelihood LMC the joint variational posterior over the
# stacked inducing vector u = [u_1; …; u_Q] has a closed-form optimum
# (collapsed bound). We compute it directly via a (QM)×(QM) linear solve, so
# there is no inner gradient loop — the variational distribution is exact
# given the prior hyperparameters (ℓs, σ²s, W, R) inherited from SS-/KM-LMC.
#
# Reuses _matern32_kernel, _pairwise_distances, _lmc_self_variance from
# `baseline.jl` (loaded earlier in `RxBayesOpt.jl`).

"""
    SVGPState <: AbstractBOState

State for the LMC sparse variational GP baseline.

# Fields
- `W::Matrix{Float64}`         : (D × Q) mixing matrix (matches SS/KM-LMC)
- `ℓs::Vector{Float64}`        : length-scales per latent
- `σ2s::Vector{Float64}`       : signal variances per latent
- `R_diag::Vector{Float64}`    : per-output observation-noise variance (D)
- `M::Int`                     : inducing points per latent
- `Z::Matrix{Float64}`         : (M × d) shared inducing locations
- `m::Vector{Vector{Float64}}` : per-latent variational means (Q × M)
- `S::Vector{Matrix{Float64}}` : per-latent variational covariances (Q × M × M)
- `S_cross::Matrix{Float64}`   : full joint (QM × QM) variational covariance
- `m_joint::Vector{Float64}`   : stacked variational mean (QM)
- `fitted::Bool`               : whether `_fit_svgp!` has been called
- `dist_norm::Float64`         : input distance normalization (matches SS/KM-LMC)
- `Xo::Matrix{Float64}`        : candidate inputs (N × d)
- `Y::Vector{Union{Missing, Vector{Float64}}}` : observations (standardized)
- `mask::BitMatrix`            : (N × D) per-cell observation mask
- `μy::Vector{Float64}`        : standardization mean
- `σy::Vector{Float64}`        : standardization std
"""
mutable struct SVGPState <: AbstractBOState
    W::Matrix{Float64}
    ℓs::Vector{Float64}
    σ2s::Vector{Float64}
    R_diag::Vector{Float64}
    M::Int
    Z::Matrix{Float64}
    m::Vector{Vector{Float64}}
    S::Vector{Matrix{Float64}}
    S_cross::Matrix{Float64}
    m_joint::Vector{Float64}
    fitted::Bool
    dist_norm::Float64
    Xo::Matrix{Float64}
    Y::Vector{Union{Missing, Vector{Float64}}}
    mask::BitMatrix
    μy::Vector{Float64}
    σy::Vector{Float64}
end

"""
    _kmeans_init(X, M; max_iter=20, rng) -> Matrix{Float64}

Pick `M` inducing locations by Lloyd's K-means clustering of the rows of `X`.
Initialized as a random subsample (Forgy method), iterated until cluster
assignments stop changing or `max_iter` is reached. Empty clusters keep
their previous centre.

Cost is O(N · M · d) per iteration — sub-millisecond for the (N, M, d) sizes
in our experiments — and is included in the SVGP wall-clock budget.
"""
function _kmeans_init(X::AbstractMatrix, M::Int;
                      max_iter::Int=20,
                      rng::AbstractRNG=MersenneTwister(0))
    N, d = size(X)
    M_use = min(M, N)
    idx = randperm(rng, N)[1:M_use]
    Z = Matrix{Float64}(X[idx, :])

    assignments = zeros(Int, N)
    for _ in 1:max_iter
        changed = false
        @inbounds for i in 1:N
            best_d = Inf
            best_j = 1
            for j in 1:M_use
                dij = 0.0
                for k in 1:d
                    dij += (X[i, k] - Z[j, k])^2
                end
                if dij < best_d
                    best_d = dij
                    best_j = j
                end
            end
            if assignments[i] != best_j
                assignments[i] = best_j
                changed = true
            end
        end

        new_Z = zeros(M_use, d)
        counts = zeros(Int, M_use)
        @inbounds for i in 1:N
            j = assignments[i]
            counts[j] += 1
            for k in 1:d
                new_Z[j, k] += X[i, k]
            end
        end
        @inbounds for j in 1:M_use
            if counts[j] > 0
                for k in 1:d
                    new_Z[j, k] /= counts[j]
                end
            else
                # Keep the previous centre for empty clusters.
                for k in 1:d
                    new_Z[j, k] = Z[j, k]
                end
            end
        end
        Z = new_Z
        changed || break
    end
    Z
end

"""
    _matern32_kmat(X1, X2, ℓ, σ2, dist_norm) -> Matrix{Float64}

Matérn 3/2 kernel matrix between rows of X1 and X2, with input distances
normalized by `dist_norm` (so `ℓ` has the same meaning as in SS/KM-LMC).
"""
function _matern32_kmat(X1::AbstractMatrix, X2::AbstractMatrix,
                        ℓ::Float64, σ2::Float64, dist_norm::Float64)
    n1, n2 = size(X1, 1), size(X2, 1)
    K = zeros(n1, n2)
    @inbounds for j in 1:n2, i in 1:n1
        r = sqrt(sqdist(row(X1, i), row(X2, j))) / dist_norm
        K[i, j] = _matern32_kernel(r, ℓ, σ2)
    end
    K
end

"""
    setup_svgp(cfg, setup_data; M=64, mask=nothing, Z_seed=0) -> SVGPState

Construct an `SVGPState` from `setup_experiment` output. The inducing locations
`Z` are a random subset of `setup_data.Xo` of size `M`. If `mask` is omitted,
the full N×D mask is assumed (all outputs observable at every point).
"""
function setup_svgp(cfg::ExperimentConfig, setup_data;
                    M::Int=64, mask::Union{Nothing, BitMatrix}=nothing,
                    Z_seed::Int=0)
    Xo = copy(setup_data.Xo)
    N = size(Xo, 1)
    M_use = min(M, N)
    Z = _kmeans_init(Xo, M_use; rng=MersenneTwister(Z_seed))

    Q = length(cfg.ℓs)
    m = [zeros(M_use) for _ in 1:Q]
    S = [Matrix{Float64}(I(M_use)) for _ in 1:Q]
    S_cross = Matrix{Float64}(I(Q * M_use))
    m_joint = zeros(Q * M_use)
    R_diag = fill(cfg.R_diag_init, cfg.D)

    mask_use = mask === nothing ? trues(N, cfg.D) : mask

    SVGPState(copy(setup_data.W), copy(cfg.ℓs), copy(cfg.σ2s), R_diag,
              M_use, Z, m, S, S_cross, m_joint, false,
              setup_data.dist_norm, Xo,
              copy(setup_data.Y), mask_use, copy(setup_data.μy),
              copy(setup_data.σy))
end

# Collect (i, d) pairs that are currently observed (Y[i] not missing AND mask[i,d]).
function _svgp_obs_pairs(Y, mask::BitMatrix)
    N, D = size(mask)
    out = Tuple{Int,Int}[]
    @inbounds for i in 1:N
        ismissing(Y[i]) && continue
        for d in 1:D
            mask[i, d] && push!(out, (i, d))
        end
    end
    out
end

"""
    _fit_svgp!(state::SVGPState) -> state

Closed-form joint optimal variational distribution q(u) for the LMC-SVGP under
Gaussian likelihood. Solves the dual system over stacked u = [u_1; …; u_Q]:

    A   = K_MM_block⁻¹ + Hᵀ Σ⁻¹ H        ((QM) × (QM))
    S*  = A⁻¹
    m*  = S* Hᵀ Σ⁻¹ y

where for each observed (i, d):
    H[k, (q-1)M+1 : qM] = W[d, q] · (K_MM_q⁻¹ k_q(Z, x_i))ᵀ
    Σ[k, k]             = R_d + Σ_q W[d,q]² · (σ²_q - k_q(x_i, Z) K_MM_q⁻¹ k_q(Z, x_i))

The per-latent marginals (m_q, S_q) and the full S_cross are stored in `state`.
"""
function _fit_svgp!(state::SVGPState)
    M = state.M
    Q = length(state.ℓs)
    Z = state.Z

    # K_MM,q (with jitter) and Cholesky per latent
    K_MM = [_matern32_kmat(Z, Z, state.ℓs[q], state.σ2s[q], state.dist_norm) +
            1e-6 * Matrix{Float64}(I(M)) for q in 1:Q]
    L_MM = [cholesky(Symmetric(K_MM[q])) for q in 1:Q]

    obs_pairs = _svgp_obs_pairs(state.Y, state.mask)
    n_obs = length(obs_pairs)
    if n_obs == 0
        # No data yet — variational posterior equals the prior.
        for q in 1:Q
            fill!(state.m[q], 0.0)
            state.S[q] = Matrix(K_MM[q])
        end
        state.S_cross = zeros(Q * M, Q * M)
        for q in 1:Q
            state.S_cross[(q-1)*M+1:q*M, (q-1)*M+1:q*M] .= K_MM[q]
        end
        fill!(state.m_joint, 0.0)
        state.fitted = true
        return state
    end

    # K_MM⁻¹ k_q(Z, x_i) for every unique observed input
    obs_pts = unique([p[1] for p in obs_pairs])
    pt_idx = Dict(p => k for (k, p) in enumerate(obs_pts))
    X_obs = state.Xo[obs_pts, :]

    K_NM = [_matern32_kmat(X_obs, Z, state.ℓs[q], state.σ2s[q], state.dist_norm)
            for q in 1:Q]                        # n_pts × M
    Φ = [Matrix((L_MM[q] \ K_NM[q]')') for q in 1:Q]   # n_pts × M

    # Assemble H, y_vec, Σ_diag
    H = zeros(n_obs, Q * M)
    y_vec = zeros(n_obs)
    Σ_diag = zeros(n_obs)
    @inbounds for (k, (i, d)) in enumerate(obs_pairs)
        pk = pt_idx[i]
        for q in 1:Q
            @views H[k, (q-1)*M+1:q*M] .= state.W[d, q] .* Φ[q][pk, :]
        end
        corr = 0.0
        for q in 1:Q
            kii = state.σ2s[q]                       # matern32(0) = σ²_q
            kiz_phi = dot(view(K_NM[q], pk, :), view(Φ[q], pk, :))
            corr += state.W[d, q]^2 * max(kii - kiz_phi, 0.0)
        end
        y_vec[k] = state.Y[i][d]
        Σ_diag[k] = state.R_diag[d] + corr
    end

    # K_MM_block⁻¹ (block-diagonal)
    K_MM_inv_block = zeros(Q * M, Q * M)
    for q in 1:Q
        K_MM_inv_block[(q-1)*M+1:q*M, (q-1)*M+1:q*M] .= inv(L_MM[q])
    end

    Σ_inv = Diagonal(1.0 ./ Σ_diag)
    HtΣinv = H' * Σ_inv
    A = K_MM_inv_block .+ HtΣinv * H
    A_sym = Symmetric(A + 1e-8 * Matrix{Float64}(I(Q * M)))
    A_chol = cholesky(A_sym)
    S_full = Matrix(inv(A_chol))
    m_full = S_full * (HtΣinv * y_vec)

    state.S_cross = S_full
    state.m_joint = m_full
    @inbounds for q in 1:Q
        idx = (q-1)*M+1:q*M
        state.m[q] = m_full[idx]
        state.S[q] = S_full[idx, idx]
    end
    state.fitted = true
    return state
end

"""
    _lmc_svgp_predict(state::SVGPState) -> (; μ_pred, σ_pred)

Predictive (mean, std) per output at every Xo[i] (standardized scale, latent
function — does NOT include observation noise R, matching `_lmc_predict`).

    μ_d(x*) = Σ_q W[d,q] · α_q(x*)ᵀ m_q                α_q(x*) = K_MM_q⁻¹ k_q(Z, x*)
    var_d(x*) = Σ_q W[d,q]² · (σ²_q - α_q(x*)ᵀ k_q(Z,x*) + α_q(x*)ᵀ S_qq α_q(x*))
              + Σ_{q≠q'} W[d,q] W[d,q'] · α_q(x*)ᵀ S_{q q'} α_{q'}(x*)
"""
function _lmc_svgp_predict(state::SVGPState)
    state.fitted || _fit_svgp!(state)

    M = state.M
    Q = length(state.ℓs)
    D = size(state.W, 1)
    N = size(state.Xo, 1)
    Z = state.Z

    K_MM = [_matern32_kmat(Z, Z, state.ℓs[q], state.σ2s[q], state.dist_norm) +
            1e-6 * Matrix{Float64}(I(M)) for q in 1:Q]
    L_MM = [cholesky(Symmetric(K_MM[q])) for q in 1:Q]
    K_NM = [_matern32_kmat(state.Xo, Z, state.ℓs[q], state.σ2s[q], state.dist_norm)
            for q in 1:Q]                            # N × M per latent
    α = [Matrix((L_MM[q] \ K_NM[q]')') for q in 1:Q] # N × M; α[q][i,:] = K_MM_q⁻¹ k_q(Z,x_i)

    μ_pred = Vector{Vector{Float64}}(undef, N)
    σ_pred = Vector{Vector{Float64}}(undef, N)

    μ_q_i = zeros(Q)
    var_q_i = zeros(Q)
    @inbounds for i in 1:N
        for q in 1:Q
            μ_q_i[q] = dot(view(α[q], i, :), state.m[q])
            kii = state.σ2s[q]
            prior_red = dot(view(α[q], i, :), view(K_NM[q], i, :))
            v_post = dot(view(α[q], i, :), state.S[q] * view(α[q], i, :))
            var_q_i[q] = max(kii - prior_red + v_post, 0.0)
        end

        μ_d = zeros(D)
        var_d = zeros(D)
        for d in 1:D
            for q in 1:Q
                μ_d[d] += state.W[d, q] * μ_q_i[q]
                var_d[d] += state.W[d, q]^2 * var_q_i[q]
            end
            # Cross-latent posterior covariance term
            for qa in 1:Q, qb in 1:Q
                qa == qb && continue
                ia = (qa - 1) * M + 1 : qa * M
                ib = (qb - 1) * M + 1 : qb * M
                Sblock = view(state.S_cross, ia, ib)
                var_d[d] += state.W[d, qa] * state.W[d, qb] *
                            dot(view(α[qa], i, :), Sblock * view(α[qb], i, :))
            end
        end
        μ_pred[i] = μ_d
        σ_pred[i] = sqrt.(max.(var_d, 1e-10))
    end

    (; μ_pred, σ_pred)
end

"""
    svgp_ucb_acquisition(state::SVGPState, cfg::ExperimentConfig)
        -> (; ucb, μ_pred, σ_pred, μs, σs)

UCB acquisition from the SVGP predictions, signature-compatible with
`baseline_ucb_acquisition` so the BO loop can be templated.
"""
function svgp_ucb_acquisition(state::SVGPState, cfg::ExperimentConfig)
    N = size(state.Xo, 1)
    D = cfg.D
    s = cfg.s

    pred = _lmc_svgp_predict(state)

    μ_pred = [pred.μ_pred[i] .* state.σy .+ state.μy for i in 1:N]
    σ_pred = [pred.σ_pred[i] .* state.σy for i in 1:N]

    μs = [dot(s, μ_pred[i]) for i in 1:N]
    σs = [sqrt(sum((s[j] * σ_pred[i][j])^2 for j in 1:D)) for i in 1:N]
    ucb = μs .+ cfg.β .* σs

    (; ucb, μ_pred, σ_pred, μs, σs)
end

"""
    run_bo_svgp!(cfg, eval_fn; svgp_state, Ytrue) -> (; result, frames)

BO loop using the SVGP surrogate. Refits the variational posterior at every
step via the closed-form solve in `_fit_svgp!`. Mirrors `run_bo_baseline!`.
"""
function run_bo_svgp!(cfg::ExperimentConfig, eval_fn;
                     svgp_state::SVGPState, Ytrue)
    Xo = svgp_state.Xo
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
                svgp_state.fitted = false  # force refit with current Y
                acq = svgp_ucb_acquisition(svgp_state, cfg)
            catch e
                @warn "SVGP prediction failed at step $step, stopping early" exception=e
                done = true
                @goto next_step
            end

            k = select_next_point(acq.ucb, svgp_state.Y)
            if k == 0
                done = true
            else
                if cfg.animate
                    plt = plot_bo_step(step, k, svgp_state, acq, Ytrue, cfg)
                    push!(frames, plt)
                end

                y_new = eval_fn(row(Xo, k))
                svgp_state.Y[k] = (y_new .- svgp_state.μy) ./ svgp_state.σy

                n_completed = step
                best = _current_best(svgp_state, cfg)
                n_obs = length(findall(!ismissing, svgp_state.Y))

                push!(best_value_history, best.value)
                push!(n_observed_history, n_obs)
                push!(R_diag_history, copy(svgp_state.R_diag))

                if step % cfg.log_every == 0
                    @info "SVGP Step $step/$(cfg.steps)" n_observed=n_obs best_value=round(best.value; digits=4)
                end
            end
            @label next_step
        end
        push!(step_times, t_step)
        done && break
    end

    best = _current_best(svgp_state, cfg)
    observed = findall(!ismissing, svgp_state.Y)
    R_learned = diagm(svgp_state.R_diag)

    result = BOResult(best.index, best.value, best.y, observed, n_completed,
                      R_learned, best_value_history, n_observed_history,
                      R_diag_history, step_times, "svgp-lmc")
    (; result, frames)
end
