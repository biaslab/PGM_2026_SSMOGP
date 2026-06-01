# Raw (RxInfer-free) Kalman filter + RTS smoother for the additive multi-output
# state-space LMC GP. Same blocks as `additive_gp_po` (`src/model.jl`); fed by
# `additive_multioutput_blocks_from_Δ` (`src/statespace.jl`).
#
# Used as the fair scalability comparison alongside KM-LMC and SVGP-LMC, and as
# a numerical diagnostic against the RxInfer implementation.

@inline _symmetrize!(M) = (M .= (M .+ M') ./ 2; M)

# Cholesky with diagonal-jitter fallback. Used by the RTS smoother to solve
# G = P_post · A' · P_pred⁻¹ without crashing on borderline-PSD P_pred.
function _safe_chol(M::AbstractMatrix; jitters=(1e-12, 1e-10, 1e-8, 1e-6))
    n = size(M, 1)
    for j in jitters
        try
            return cholesky(Symmetric(M + j * I(n)))
        catch
            continue
        end
    end
    cholesky(Symmetric(M + 1e-4 * I(n)))   # last resort
end

"""
    ss_lmc_filter_smooth(P, A, Q, H, τ, Y_flat, N, D) -> (; μ_pred, σ_pred)

Kalman filter + RTS smoother for the additive multi-output state-space LMC GP.
Mirrors `additive_gp_po` (`src/model.jl`) exactly:

- State `f[i] ∈ R^(2Q)`, prior `f[0] ~ N(0, P)`
- Transition `f[i] = A[i] · f[i-1] + w[i]`, `w[i] ~ N(0, Q[i])`
- Per-output observation `Y[(i-1)D + d] ~ N(H[d,:]ᵀ · f[i], 1/τ[d])`, with
  `missing` entries skipped

# Inputs
- `P::Matrix`: stationary prior covariance (2Q × 2Q)
- `A::Vector{Matrix}`: per-step transition matrices, length `N`
- `Q::Vector{Matrix}`: per-step process noise covariances, length `N`
- `H::Matrix`: observation matrix (D × 2Q)
- `τ::Vector{Float64}`: per-output observation precisions, length `D`
- `Y_flat::Vector{Union{Missing, Float64}}`: flat observations, length `N*D`,
  indexed point-major (`Y_flat[(i-1)D + d]`)
- `N::Int`, `D::Int`: chain length and output count

# Returns
- `μ_pred::Vector{Vector{Float64}}`: length `N`, each a D-vector — smoothed
  posterior mean of `H·f[i]` on the standardized output scale
- `σ_pred::Vector{Vector{Float64}}`: length `N`, each a D-vector — posterior
  standard deviation of `H·f[i]` (latent function only; no observation noise)

Updates are done as **sequential scalar measurements** (one per observed
(i, d) pair). This avoids any D×D innovation-matrix inversion and makes
per-cell masking trivial. Joseph form is used for the covariance update so
the posterior covariance stays numerically symmetric and PSD.
"""
function ss_lmc_filter_smooth(P::AbstractMatrix,
                              A::AbstractVector,
                              Q::AbstractVector,
                              H::AbstractMatrix,
                              τ::AbstractVector,
                              Y_flat::AbstractVector,
                              N::Int, D::Int)
    n = size(P, 1)
    @assert size(P, 2) == n
    @assert size(H, 2) == n
    @assert size(H, 1) == D
    @assert length(A) == N
    @assert length(Q) == N
    @assert length(τ) == D
    @assert length(Y_flat) == N * D

    # Pre-extract h_d as plain Vector{Float64} for fast dot products.
    h = [Vector{Float64}(H[d, :]) for d in 1:D]
    R_diag = [1.0 / τ[d] for d in 1:D]

    # Storage for filter quantities at every step.
    m_post = [zeros(n) for _ in 1:N]
    P_post = [zeros(n, n) for _ in 1:N]
    m_pred = [zeros(n) for _ in 1:N]
    P_pred = [zeros(n, n) for _ in 1:N]

    # Initial state: m = 0, Σ = P.  Predict from step 0 to step 1.
    m_prev = zeros(n)
    P_prev = Matrix{Float64}(P)
    _symmetrize!(P_prev)

    for i in 1:N
        # ── Predict ──────────────────────────────────────────────────────
        Ai = A[i]; Qi = Q[i]
        mp = Ai * m_prev
        Pp = Ai * P_prev * Ai' + Qi
        _symmetrize!(Pp)
        m_pred[i] = mp
        P_pred[i] = copy(Pp)

        # ── Sequential scalar updates over observed (i, d) ───────────────
        m = mp
        Pm = Pp
        for d in 1:D
            yk = Y_flat[(i - 1) * D + d]
            ismissing(yk) && continue

            hd = h[d]
            Phd = Pm * hd                        # (n,)
            S = dot(hd, Phd) + R_diag[d]         # scalar
            S > 0 || (S = R_diag[d])             # safety; should never trigger
            K = Phd ./ S                         # (n,)
            r = yk - dot(hd, m)
            m = m .+ K .* r

            # Joseph form: (I - K hᵀ) P (I - K hᵀ)ᵀ + K Kᵀ / τ
            #   = Pm - K (hᵀ Pm) - (Pm h) Kᵀ + K (hᵀ Pm h + 1/τ) Kᵀ
            #   = Pm - K Phd' - Phd K' + S * K K'
            Pm = Pm .- K .* Phd' .- Phd .* K' .+ S .* (K * K')
            _symmetrize!(Pm)
        end

        m_post[i] = m
        P_post[i] = Pm
        m_prev, P_prev = m, Pm
    end

    # ── Backward RTS smoother ────────────────────────────────────────────
    m_s = [copy(m_post[i]) for i in 1:N]
    P_s = [copy(P_post[i]) for i in 1:N]
    for i in (N - 1):-1:1
        # G = P_post[i] · A[i+1]' · inv(P_pred[i+1])
        Lp = _safe_chol(P_pred[i + 1])
        # Solve Lp · (Lp' · X') = (P_post[i] · A[i+1]')'   for X.
        rhs = (P_post[i] * A[i + 1]')'
        G = (Lp \ rhs)'

        m_s[i] = m_post[i] .+ G * (m_s[i + 1] .- m_pred[i + 1])
        P_s[i] = P_post[i] .+ G * (P_s[i + 1] .- P_pred[i + 1]) * G'
        _symmetrize!(P_s[i])
    end

    # ── Per-position output predictions ──────────────────────────────────
    μ_pred = Vector{Vector{Float64}}(undef, N)
    σ_pred = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        μ_pred[i] = H * m_s[i]
        # Latent function variance: diag(H · P_s[i] · H')
        HP = H * P_s[i]
        v = [dot(view(HP, d, :), view(H, d, :)) for d in 1:D]
        σ_pred[i] = sqrt.(max.(v, 1e-12))
    end

    (; μ_pred, σ_pred)
end
