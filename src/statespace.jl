"""
    matern32_blocks_from_Δ(Δ; ℓ=1.0, σ2=1.0) -> (; A, Q, P∞, H)

Discretize the Matérn 3/2 SDE into state-space matrices given inter-point
distances `Δ`.

The continuous-time SDE for a Matérn 3/2 process with length-scale `ℓ` and
variance `σ2` has state dimension 2. Given distances `Δ[i]` between
consecutive points in a chain, this returns:
- `A[i]`: transition matrices `exp(F·Δ[i])`
- `Q[i]`: process noise covariances `P∞ - A[i]·P∞·A[i]'`
- `P∞`: stationary covariance
- `H`: observation vector `[1, 0]`
"""
function matern32_blocks_from_Δ(Δ::AbstractVector; ℓ=1.0, σ2=1.0)
    λ = sqrt(3) / ℓ
    F = [0.0 1.0; -λ^2 -2λ]
    P∞ = [σ2 0.0; 0.0 λ^2*σ2]
    A = [exp(F * Δ[i]) for i in eachindex(Δ)]
    Q = Vector{Matrix{Float64}}(undef, length(Δ))
    for i in eachindex(Δ)
        Qi = P∞ - A[i] * P∞ * A[i]'
        Qi = (Qi + Qi') / 2  # symmetrize
        # Clamp negative eigenvalues to enforce PSD
        eig = eigen(Symmetric(Qi))
        eig.values .= max.(eig.values, 1e-12)
        Q[i] = eig.vectors * Diagonal(eig.values) * eig.vectors'
    end
    H = [1.0, 0.0]
    (; A, Q, P∞, H)
end

"""
    additive_multioutput_blocks_from_Δ(Δ; ℓs, σ2s, W) -> (; A, Q, P, H)

Stack `Q` independent latent Matérn 3/2 state-space GPs into a block-diagonal
model and apply mixing matrix `W` (D × Q) to produce D outputs.

This implements the Linear Model of Coregionalization (LMC) in state-space form:
- Each latent GP `q` has its own length-scale `ℓs[q]` and variance `σ2s[q]`
- Transition and noise matrices are block-diagonal across latent GPs
- The observation matrix `H` applies `W` to mix latent states into outputs
"""
function additive_multioutput_blocks_from_Δ(Δ; ℓs, σ2s, W::AbstractMatrix)
    Q = length(ℓs)
    @assert Q == length(σ2s)
    D, QW = size(W)
    @assert QW == Q
    lat = [matern32_blocks_from_Δ(Δ; ℓ=ℓs[q], σ2=σ2s[q]) for q in 1:Q]
    A_big = [blockdiag((lat[q].A[i] for q in 1:Q)...) for i in eachindex(Δ)]
    Q_big = [Matrix(Hermitian(blockdiag((lat[q].Q[i] for q in 1:Q)...))) for i in eachindex(Δ)]
    P_big = Matrix(Hermitian(blockdiag((lat[q].P∞ for q in 1:Q)...)))
    H_big = hcat([W[:, q] * lat[q].H' for q in 1:Q]...)
    (; A=A_big, Q=Q_big, P=P_big, H=H_big)
end
