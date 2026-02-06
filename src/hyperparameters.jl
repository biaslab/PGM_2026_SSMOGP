"""
    log_marginal_likelihood(Y_obs, P, A, Q, H, R) -> Float64

Compute the log marginal likelihood via a Kalman filter forward pass.

Runs the prediction-update recursion over all `N` time steps, accumulating
the log-likelihood of observed (non-missing) entries. Missing observations
are skipped (prediction only, no update).

# Arguments
- `Y_obs`: length-N vector of observations (or `missing`)
- `P`: initial state covariance
- `A`: length-N vector of transition matrices
- `Q`: length-N vector of process noise covariances
- `H`: observation matrix
- `R`: observation noise covariance
"""
function log_marginal_likelihood(Y_obs, P, A, Q, H, R)
    N = length(Y_obs)
    n = size(P, 1)
    d = size(R, 1)

    μ = zeros(n)
    Σ = copy(P)
    ll = 0.0

    for i in 1:N
        ismissing(Y_obs[i]) && continue

        μ_pred = H * A[i] * μ
        Σ_pred = H * (A[i] * Σ * A[i]' + Q[i]) * H' + R
        Σ_pred = 0.5 * (Σ_pred + Σ_pred')

        y = Y_obs[i]
        r = y - μ_pred
        ll -= 0.5 * (logdet(Σ_pred) + dot(r, Σ_pred \ r) + d * log(2π))

        K = (A[i] * Σ * A[i]' + Q[i]) * H' / Σ_pred
        μ = A[i] * μ + K * r
        Σ = A[i] * Σ * A[i]' + Q[i] - K * Σ_pred * K'
    end

    ll
end

"""
    tune_hyperparameters(Y_obs, Δ, W_init, D, Q; R_diag_init=0.2)

Optimize GP hyperparameters via Type-II maximum likelihood (LBFGS).

Jointly optimizes log-length-scales, log-variances, mixing matrix entries,
and log-observation-noise. Returns a named tuple `(ℓs, σ2s, W, R_diag)`.
"""
function tune_hyperparameters(Y_obs, Δ, W_init, D, Q; R_diag_init=0.2)
    function objective(θ)
        ℓs = exp.(θ[1:Q])
        σ2s = exp.(θ[Q+1:2Q])
        W_vec = θ[2Q+1:2Q+D*Q]
        W = reshape(W_vec, D, Q)
        R_diag = exp(θ[end])

        blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs, σ2s, W)
        R = Diagonal(fill(R_diag, D))

        -log_marginal_likelihood(Y_obs, blocks.P, blocks.A, blocks.Q, blocks.H, R)
    end

    θ0 = vcat(
        zeros(Q),
        log.(fill(4.0, Q)),
        vec(W_init),
        log(R_diag_init)
    )

    result = optimize(objective, θ0, LBFGS(), Optim.Options(iterations=50, show_trace=false))
    θ_opt = Optim.minimizer(result)

    ℓs_opt = exp.(θ_opt[1:Q])
    σ2s_opt = exp.(θ_opt[Q+1:2Q])
    W_opt = reshape(θ_opt[2Q+1:2Q+D*Q], D, Q)
    R_diag_opt = exp(θ_opt[end])

    (; ℓs=ℓs_opt, σ2s=σ2s_opt, W=W_opt, R_diag=R_diag_opt)
end
