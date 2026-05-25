"""
    row(X, i) -> view

Zero-allocation view of the `i`-th row of matrix `X`.
"""
@inline row(X::AbstractMatrix, i::Integer) = view(X, i, :)

"""
    sqdist(a, b) -> Float64

Squared Euclidean distance between vectors `a` and `b`, computed with SIMD.
"""
@inline function sqdist(a::AbstractVector, b::AbstractVector)
    s = zero(promote_type(eltype(a), eltype(b)))
    @inbounds @simd for j in eachindex(a, b)
        d = a[j] - b[j]
        s += d * d
    end
    s
end

"""
    nn_chain_order(X) -> Vector{Int}

Greedy nearest-neighbor chain ordering of the rows of `X`.

Starting from the first row, repeatedly visits the closest unvisited row
(by squared Euclidean distance). Returns a permutation vector of row indices.
This heuristic enables state-space GP inference on high-dimensional inputs
by arranging points into a 1D chain.
"""
function nn_chain_order(X::AbstractMatrix{<:Real})
    N = size(X, 1)
    remaining = collect(1:N)
    order = Vector{Int}(undef, N)
    order[1] = popfirst!(remaining)
    for i in 2:N
        last = order[i-1]
        j = argmin([sqdist(row(X, k), row(X, last)) for k in remaining])
        order[i] = remaining[j]
        splice!(remaining, j)
    end
    order
end

"""
    nn_chain_quality(Xo) -> NamedTuple

Diagnostic summary of an NN-chain ordering's "stretch": pairwise Euclidean
distances between consecutive rows of `Xo`.

In low input dimensions the chain links should be short and uniform; as `d`
grows, even the nearest unvisited neighbour drifts further and the spread of
`Δ` values widens — a direct fingerprint of the curse of dimensionality on
the heuristic.

Returns a NamedTuple with `mean_delta`, `median_delta`, `max_delta`,
`min_delta`, `total_length`, and the raw vector `Δ` (length N-1).
"""
function nn_chain_quality(Xo::AbstractMatrix{<:Real})
    N = size(Xo, 1)
    Δ = [sqrt(sqdist(row(Xo, i), row(Xo, i - 1))) for i in 2:N]
    (; mean_delta   = mean(Δ),
       median_delta = median(Δ),
       max_delta    = maximum(Δ),
       min_delta    = minimum(Δ),
       total_length = sum(Δ),
       Δ            = Δ)
end

"""
    blockdiag(mats...) -> Matrix

Construct a block-diagonal matrix from the given matrices.
"""
function blockdiag(mats::AbstractMatrix...)
    T = promote_type(map(eltype, mats)...)
    m = sum(size(M, 1) for M in mats)
    n = sum(size(M, 2) for M in mats)
    B = zeros(T, m, n)
    r, c = 1, 1
    for M in mats
        B[r:r+size(M, 1)-1, c:c+size(M, 2)-1] .= M
        r += size(M, 1)
        c += size(M, 2)
    end
    B
end
