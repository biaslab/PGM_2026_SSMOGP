# Vecchia / NNGP (LMC) baseline — Vecchia 1988; Datta et al. 2016; Katzfuss & Guinness 2021.
#
# A fourth scalable LMC-GP baseline alongside SS-LMC (state-space message passing,
# O(C)), KM-LMC (dense kernel matrix, O((D·C)³)) and SVGP-LMC (inducing points,
# O(C·M²)). Like SS-LMC it is built on a *nearest-neighbour ordering* of the
# M-dimensional inputs, which makes it the most direct competitor to the chain
# approximation this paper analyses: both replace the dense joint by a sparse
# conditional factorization keyed on nearest neighbours, but
#
#   * SS-LMC compresses the M-dimensional geometry into ONE 1-D Markov chain
#     (each point conditions on its single chain predecessor) and then runs
#     *exact* inference on that surrogate model;
#   * Vecchia/NNGP keeps the M-dimensional geometry and lets each point condition
#     on its `m` nearest predecessors, giving a variable-size, higher-order but
#     *approximate* sparse precision.
#
# Both share the LMC prior (W, ℓs, σ2s, R) with SS-LMC and KM-LMC, so any
# difference in held-out score is attributable to the sparsity structure alone.
#
# ── Which Vecchia variant ───────────────────────────────────────────────────
# We use the standard *observed-first* (a.k.a. "obs-pred") ordering of
# Katzfuss, Guinness, Gong & Zilber (2020, "Vecchia approximations of
# Gaussian-process predictions"): training locations are ordered first by
# maxmin ordering (Guinness 2018), prediction locations last. Under that
# ordering the Vecchia factorization
#
#     p(y_train) ∏_k p(f_k | y_{g(k)})
#
# assigns every prediction location a conditioning set g(k) drawn from training
# locations only, so the Vecchia predictive marginals are available in closed
# form as local kriging on the `m` nearest observed neighbours — no sparse
# selected inversion is needed and the cost is O(C_test · (mD)³), genuinely
# sparse rather than a dense solve in disguise.
#
# ── Multivariate (LMC) conditioning sets ────────────────────────────────────
# Following the general Vecchia framework, conditioning sets are formed at the
# *location* level: a point conditions on ALL observed outputs at its `m`
# nearest neighbouring locations, so a conditioning set holds up to m·D scalar
# responses. This is the natural multi-output generalization and keeps the
# comparison to SVGP-LMC honest (m·D conditioning values vs. M inducing points).
#
# Partial observability is handled the same way it is for KM-LMC: entries that
# are missing (`Y[i] === missing`) or masked out (`mask[i,d] == false`) are
# simply absent from the conditioning sets.
#
# Reuses `_matern32_kernel`, `_pairwise_distances`, `_lmc_self_variance` from
# `baseline.jl` (loaded earlier in `RxBayesOpt.jl`).

"""
    VecchiaState <: AbstractBOState

State for the Vecchia/NNGP LMC baseline.

# Fields
- `W::Matrix{Float64}`      : (D × Q) mixing matrix (shared with SS-/KM-LMC)
- `ℓs::Vector{Float64}`     : length-scales per latent
- `σ2s::Vector{Float64}`    : signal variances per latent
- `R_diag::Vector{Float64}` : per-output observation-noise variance (D)
- `m::Int`                  : number of neighbouring *locations* conditioned on
- `dist_norm::Float64`      : input distance normalization (shared with SS-/KM-LMC)
- `Xo::Matrix{Float64}`     : candidate inputs (C × M)
- `mask::BitMatrix`         : (C × D) per-cell observation mask
- `Y::Vector{Union{Missing, Vector{Float64}}}` : observations (standardized)
- `μy::Vector{Float64}`     : standardization mean
- `σy::Vector{Float64}`     : standardization std
"""
mutable struct VecchiaState <: AbstractBOState
    W::Matrix{Float64}
    ℓs::Vector{Float64}
    σ2s::Vector{Float64}
    R_diag::Vector{Float64}
    m::Int
    dist_norm::Float64
    Xo::Matrix{Float64}
    mask::BitMatrix
    Y::Vector{Union{Missing, Vector{Float64}}}
    μy::Vector{Float64}
    σy::Vector{Float64}
end

"""
    setup_vecchia(cfg, setup_data; m=20, mask=nothing) -> VecchiaState

Construct a `VecchiaState` from `setup_experiment` output, inheriting the LMC
prior (`W`, `cfg.ℓs`, `cfg.σ2s`, `cfg.R_diag_init`) and input normalization used
by SS-LMC and KM-LMC. `m` is the number of neighbouring locations each point
conditions on. If `mask` is omitted, the full C×D mask is assumed.
"""
function setup_vecchia(cfg::ExperimentConfig, setup_data;
                       m::Int=20, mask::Union{Nothing, BitMatrix}=nothing)
    Xo = copy(setup_data.Xo)
    N = size(Xo, 1)
    mask_use = mask === nothing ? trues(N, cfg.D) : mask

    VecchiaState(copy(setup_data.W), copy(cfg.ℓs), copy(cfg.σ2s),
                 fill(cfg.R_diag_init, cfg.D), m,
                 setup_data.dist_norm, Xo, mask_use,
                 copy(setup_data.Y), copy(setup_data.μy), copy(setup_data.σy))
end

# ─── Ordering and neighbour sets ────────────────────────────────────────────

"""
    _maxmin_order(X, idx) -> Vector{Int}

Maxmin ordering (Guinness 2018) of the rows of `X` restricted to `idx`.

Starts at the point closest to the centroid of `X[idx, :]`, then repeatedly
appends the remaining point whose minimum distance to the already-ordered set is
largest. The resulting sequence is coarse-to-fine, which is what makes small
Vecchia conditioning sets accurate: early points are spread over the whole
domain, so every later point has informative predecessors nearby.

Returns indices into the ORIGINAL rows of `X` (i.e. a permutation of `idx`).
Cost is O(|idx|²·M).
"""
function _maxmin_order(X::AbstractMatrix, idx::Vector{Int})
    n = length(idx)
    n == 0 && return Int[]

    centroid = vec(mean(view(X, idx, :), dims=1))
    first_pos = argmin([sqdist(row(X, i), centroid) for i in idx])

    ord = Vector{Int}(undef, n)
    ord[1] = idx[first_pos]
    remaining = copy(idx)
    deleteat!(remaining, first_pos)

    # min_dist[k] = squared distance from remaining[k] to the ordered set so far.
    min_dist = [sqdist(row(X, j), row(X, ord[1])) for j in remaining]

    for i in 2:n
        pos = argmax(min_dist)
        ord[i] = remaining[pos]
        deleteat!(remaining, pos)
        deleteat!(min_dist, pos)
        @inbounds for k in eachindex(remaining)
            dk = sqdist(row(X, remaining[k]), row(X, ord[i]))
            dk < min_dist[k] && (min_dist[k] = dk)
        end
    end
    ord
end

"""
    _m_nearest(X, i, pool, m; exclude_self=false) -> Vector{Int}

Indices of the `m` rows of `X` among `pool` that are closest to row `i`.
Ties are broken by index order. `exclude_self=true` drops `i` from the result.
"""
function _m_nearest(X::AbstractMatrix, i::Int, pool::Vector{Int}, m::Int;
                    exclude_self::Bool=false)
    isempty(pool) && return Int[]
    cand = exclude_self ? filter(!=(i), pool) : pool
    isempty(cand) && return Int[]
    m_use = min(m, length(cand))
    m_use == length(cand) && return copy(cand)

    d = [sqdist(row(X, j), row(X, i)) for j in cand]
    cand[partialsortperm(d, 1:m_use)]
end

# Locations carrying at least one observed entry.
function _observed_locations(state::VecchiaState)
    N = size(state.Xo, 1)
    [i for i in 1:N if !ismissing(state.Y[i]) && any(view(state.mask, i, :))]
end

# Flatten the observed (location, output) pairs at `locs` into parallel vectors.
function _observed_pairs_at(state::VecchiaState, locs::AbstractVector{Int})
    D = size(state.W, 1)
    pts = Int[]; dims = Int[]; vals = Float64[]
    @inbounds for j in locs
        ismissing(state.Y[j]) && continue
        for e in 1:D
            if state.mask[j, e]
                push!(pts, j); push!(dims, e); push!(vals, state.Y[j][e])
            end
        end
    end
    (; pts, dims, vals)
end

# ─── LMC covariance over (location, output) pairs ───────────────────────────

"""
    _lmc_cov_pairs(Xo, pts_a, dims_a, pts_b, dims_b, W, ℓs, σ2s, dist_norm)

Noise-free LMC covariance between two lists of (location, output) pairs:

    K[a, b] = Σ_q W[dims_a[a], q]·W[dims_b[b], q]·matern32(‖x_{pts_a[a]} - x_{pts_b[b]}‖/dist_norm, ℓ_q, σ²_q)

The per-latent kernel value depends only on the location pair, so it is computed
once per (location, location, latent) triple and reused across the D² output
combinations.
"""
function _lmc_cov_pairs(Xo::AbstractMatrix,
                        pts_a::Vector{Int}, dims_a::Vector{Int},
                        pts_b::Vector{Int}, dims_b::Vector{Int},
                        W::Matrix{Float64}, ℓs::Vector{Float64},
                        σ2s::Vector{Float64}, dist_norm::Float64)
    na, nb = length(pts_a), length(pts_b)
    Q = length(ℓs)
    K = zeros(na, nb)

    # Cache kernel values per (unique location a, unique location b, latent).
    ua = unique(pts_a); ub = unique(pts_b)
    ia = Dict(p => k for (k, p) in enumerate(ua))
    ib = Dict(p => k for (k, p) in enumerate(ub))
    kcache = Array{Float64}(undef, length(ua), length(ub), Q)
    @inbounds for (bj, jb) in enumerate(ub), (ai, ja) in enumerate(ua)
        r = sqrt(sqdist(row(Xo, ja), row(Xo, jb))) / dist_norm
        for q in 1:Q
            kcache[ai, bj, q] = _matern32_kernel(r, ℓs[q], σ2s[q])
        end
    end

    @inbounds for b in 1:nb
        eb = dims_b[b]; cb = ib[pts_b[b]]
        for a in 1:na
            da = dims_a[a]; ca = ia[pts_a[a]]
            acc = 0.0
            for q in 1:Q
                acc += W[da, q] * W[eb, q] * kcache[ca, cb, q]
            end
            K[a, b] = acc
        end
    end
    K
end

# ─── Prediction ─────────────────────────────────────────────────────────────

"""
    _lmc_vecchia_predict(state::VecchiaState) -> (; μ_pred, σ_pred)

Vecchia/NNGP predictive (mean, std) per output at every `Xo[i]`, on the
standardized scale and for the *latent* function (excludes observation noise R),
matching `_lmc_predict_po` and `_lmc_svgp_predict`.

Under the observed-first ordering, location `i` conditions on the observed
entries at its `m` nearest observed locations `g(i)`:

    μ_d(x_i)   = k_d,g(i) (K_g(i),g(i) + R_g(i))⁻¹ y_g(i)
    var_d(x_i) = Σ_q W[d,q]²σ²_q − k_d,g(i) (K_g(i),g(i) + R_g(i))⁻¹ k_g(i),d

An already-observed location includes itself in its own conditioning set, so
training-point predictions are shrunk toward their observations exactly as in
the dense baseline. Cost is O(C·(mD)³) with no dense C-scale factorization.
"""
function _lmc_vecchia_predict(state::VecchiaState)
    Xo = state.Xo
    N  = size(Xo, 1)
    D  = size(state.W, 1)
    prior_var = _lmc_self_variance(state.W, state.σ2s)

    obs_locs = _observed_locations(state)

    μ_pred = Vector{Vector{Float64}}(undef, N)
    σ_pred = Vector{Vector{Float64}}(undef, N)

    if isempty(obs_locs)
        for i in 1:N
            μ_pred[i] = zeros(D)
            σ_pred[i] = sqrt.(prior_var)
        end
        return (; μ_pred, σ_pred)
    end

    # Query pairs for a single location: all D outputs.
    q_dims = collect(1:D)

    for i in 1:N
        nbr = _m_nearest(Xo, i, obs_locs, state.m)
        cond = _observed_pairs_at(state, nbr)
        nc = length(cond.pts)

        if nc == 0
            μ_pred[i] = zeros(D)
            σ_pred[i] = sqrt.(prior_var)
            continue
        end

        K_cc = _lmc_cov_pairs(Xo, cond.pts, cond.dims, cond.pts, cond.dims,
                              state.W, state.ℓs, state.σ2s, state.dist_norm)
        @inbounds for a in 1:nc
            K_cc[a, a] += state.R_diag[cond.dims[a]]
        end

        q_pts = fill(i, D)
        K_qc = _lmc_cov_pairs(Xo, q_pts, q_dims, cond.pts, cond.dims,
                              state.W, state.ℓs, state.σ2s, state.dist_norm)

        L = cholesky(Symmetric(K_cc) + 1e-10I)
        μ_pred[i] = K_qc * (L \ cond.vals)

        V = L.L \ K_qc'                       # nc × D
        var_d = prior_var .- vec(sum(abs2, V; dims=1))
        σ_pred[i] = sqrt.(max.(var_d, 1e-10))
    end

    (; μ_pred, σ_pred)
end

"""
    vecchia_loglik(state::VecchiaState) -> Float64

Vecchia log-likelihood of the observed data under the LMC prior:

    log p̂(y) = Σ_k log p(y_{loc ord[k]} | y_{g(k)}),

with locations in maxmin order and `g(k)` the observed entries at the `m`
nearest *preceding* locations. This is the object Vecchia/NNGP methods maximize
for hyperparameter estimation. The paper fixes (W, ℓ_l, γ²_l), so this is not
optimized here; it is provided so the baseline is a complete Vecchia
implementation rather than prediction-only, and to sanity-check that the
approximation tightens as `m` grows.
"""
function vecchia_loglik(state::VecchiaState)
    Xo = state.Xo
    obs_locs = _observed_locations(state)
    isempty(obs_locs) && return 0.0

    ord = _maxmin_order(Xo, obs_locs)
    ll = 0.0

    for k in eachindex(ord)
        i = ord[k]
        self = _observed_pairs_at(state, [i])
        ns = length(self.pts)
        ns == 0 && continue

        prev = ord[1:k-1]
        nbr = _m_nearest(Xo, i, prev, state.m)
        cond = _observed_pairs_at(state, nbr)
        nc = length(cond.pts)

        K_ss = _lmc_cov_pairs(Xo, self.pts, self.dims, self.pts, self.dims,
                              state.W, state.ℓs, state.σ2s, state.dist_norm)
        @inbounds for a in 1:ns
            K_ss[a, a] += state.R_diag[self.dims[a]]
        end

        if nc == 0
            Lp = cholesky(Symmetric(K_ss) + 1e-10I)
            ll += -0.5 * (ns * log(2π) + 2 * sum(log, diag(Lp.L)) +
                          sum(abs2, Lp.L \ self.vals))
            continue
        end

        K_cc = _lmc_cov_pairs(Xo, cond.pts, cond.dims, cond.pts, cond.dims,
                              state.W, state.ℓs, state.σ2s, state.dist_norm)
        @inbounds for a in 1:nc
            K_cc[a, a] += state.R_diag[cond.dims[a]]
        end
        K_sc = _lmc_cov_pairs(Xo, self.pts, self.dims, cond.pts, cond.dims,
                              state.W, state.ℓs, state.σ2s, state.dist_norm)

        Lc = cholesky(Symmetric(K_cc) + 1e-10I)
        μ_c = K_sc * (Lc \ cond.vals)
        V = Lc.L \ K_sc'
        Σ_c = Symmetric(K_ss - V' * V) + 1e-10I

        Lp = cholesky(Σ_c)
        r = self.vals .- μ_c
        ll += -0.5 * (ns * log(2π) + 2 * sum(log, diag(Lp.L)) + sum(abs2, Lp.L \ r))
    end
    ll
end

"""
    vecchia_ucb_acquisition(state::VecchiaState, cfg::ExperimentConfig)
        -> (; ucb, μ_pred, σ_pred, μs, σs)

UCB acquisition from the Vecchia predictions, signature-compatible with
`baseline_po_ucb_acquisition` and `svgp_ucb_acquisition`.
"""
function vecchia_ucb_acquisition(state::VecchiaState, cfg::ExperimentConfig)
    N = size(state.Xo, 1)
    D = cfg.D
    s = cfg.s

    pred = _lmc_vecchia_predict(state)
    μ_pred = [pred.μ_pred[i] .* state.σy .+ state.μy for i in 1:N]
    σ_pred = [pred.σ_pred[i] .* state.σy for i in 1:N]

    μs = [dot(s, μ_pred[i]) for i in 1:N]
    σs = [sqrt(sum((s[j] * σ_pred[i][j])^2 for j in 1:D)) for i in 1:N]
    ucb = μs .+ cfg.β .* σs

    (; ucb, μ_pred, σ_pred, μs, σs)
end

"""
    vecchia_variance_acquisition(state::VecchiaState, cfg::ExperimentConfig)
        -> (; score, μ_pred, σ_pred)

Pure-variance acquisition, signature-compatible with
`baseline_po_variance_acquisition` (used by the sequential-design and
dim-sweep drivers). Predictions are returned on the original output scale.
"""
function vecchia_variance_acquisition(state::VecchiaState, cfg::ExperimentConfig)
    N = size(state.Xo, 1)

    pred = _lmc_vecchia_predict(state)
    μ_pred = [pred.μ_pred[i] .* state.σy .+ state.μy for i in 1:N]
    σ_pred = [pred.σ_pred[i] .* state.σy for i in 1:N]

    score = [sum(σ_pred[i] .^ 2) for i in 1:N]
    (; score, μ_pred, σ_pred)
end
