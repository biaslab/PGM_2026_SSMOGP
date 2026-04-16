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
    nn_chain_order(X; start_idx=1) -> Vector{Int}

Greedy nearest-neighbor chain ordering of the rows of `X`.

Starting from row `start_idx`, repeatedly visits the closest unvisited row
(by squared Euclidean distance). Returns a permutation vector of row indices.
This heuristic enables state-space GP inference on high-dimensional inputs
by arranging points into a 1D chain.
"""
function nn_chain_order(X::AbstractMatrix{<:Real}; start_idx::Int=1)
    N = size(X, 1)
    remaining = collect(1:N)
    order = Vector{Int}(undef, N)
    idx_pos = findfirst(==(start_idx), remaining)
    order[1] = splice!(remaining, idx_pos)
    for i in 2:N
        last = order[i-1]
        j = argmin([sqdist(row(X, k), row(X, last)) for k in remaining])
        order[i] = remaining[j]
        splice!(remaining, j)
    end
    order
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
