# Partial-observation experiment: demonstrates FFG modularity.
#
# The state-space GP naturally handles missing per-output observations via
# message passing — each scalar observation factor is independent in the FFG.
# The kernel-matrix baseline must restructure its covariance matrix to handle
# the variable-size observation set.

"""
    _generate_obs_mask(cfg::ExperimentConfig, N::Int, rng) -> BitMatrix

Generate an N × D observation mask.

Patterns:
- `:full` — all outputs observed at every point
- `:sensor_groups` — even-indexed points observe outputs in group 1,
  odd-indexed points observe outputs in group 2 (alternating sensor groups)
"""
function _generate_obs_mask(cfg::ExperimentConfig, N::Int, rng)
    D = cfg.D
    mask = BitMatrix(undef, N, D)

    if cfg.obs_pattern == :full
        fill!(mask, true)
    elseif cfg.obs_pattern == :sensor_groups
        mid = div(D, 2)
        for i in 1:N
            if iseven(i)
                mask[i, :] .= [d <= mid for d in 1:D]
            else
                mask[i, :] .= [d > mid for d in 1:D]
            end
        end
    else
        error("Unknown obs_pattern: $(cfg.obs_pattern)")
    end

    mask
end

"""
    _build_Y_flat(Y, mask, N, D) -> Vector{Union{Missing, Float64}}

Build the flat N*D observation vector from point-level Y and mask.
Entry (i-1)*D + d is Y[i][d] if Y[i] is not missing and mask[i,d] is true,
otherwise missing.
"""
function _build_Y_flat(Y::Vector{Union{Missing, Vector{Float64}}}, mask::BitMatrix, N::Int, D::Int)
    Y_flat = Vector{Union{Missing, Float64}}(missing, N * D)
    for i in 1:N
        ismissing(Y[i]) && continue
        for d in 1:D
            if mask[i, d]
                Y_flat[(i-1)*D + d] = Y[i][d]
            end
        end
    end
    Y_flat
end

"""
    setup_po(cfg::ExperimentConfig, setup_data) -> POState

Create a POState from the output of `setup_experiment`.

Uses fixed per-output noise precisions τ = 1/R_diag_init (no Wishart).
Generates the observation mask and builds the flat Y vector.
"""
function setup_po(cfg::ExperimentConfig, setup_data)
    D = cfg.D
    N = cfg.N
    rng = MersenneTwister(cfg.seed + 1000)

    W_copy = copy(setup_data.W)
    blocks = setup_data.blocks
    Y_copy = copy(setup_data.Y)
    μy_copy = copy(setup_data.μy)
    σy_copy = copy(setup_data.σy)

    τ = fill(1.0 / cfg.R_diag_init, D)
    e_vecs = [Float64.(I(D)[:, d]) for d in 1:D]

    mask = _generate_obs_mask(cfg, N, rng)
    Y_flat = _build_Y_flat(Y_copy, mask, N, D)

    POState(blocks, W_copy, τ, e_vecs, mask, Y_copy, Y_flat, μy_copy, σy_copy)
end

"""
    run_bo_po!(cfg::ExperimentConfig, eval_fn; po_state, Xo, Δ, Ytrue) -> (; result, frames)

Run BO loop with the partial-observation state-space GP model.

At each step:
1. Run RxInfer inference with `additive_gp_po` (single pass, no VMP iterations)
2. Compute UCB from `my[i]` posteriors (D-vector)
3. Select next point, evaluate, store full D-vector in Y[k],
   fill Y_flat only where mask[k,d] is true
"""
function run_bo_po!(cfg::ExperimentConfig, eval_fn; po_state::POState, Xo, Δ, Ytrue)
    N = cfg.N
    D = cfg.D
    frames = cfg.animate ? [] : nothing

    best_value_history = Float64[]
    n_observed_history = Int[]
    R_diag_history = Vector{Float64}[]
    step_times = Float64[]

    n_completed = 0
    done = false
    for step in 1:cfg.steps
        t_step = @elapsed begin
            local res
            try
                res = infer(
                    model=additive_gp_po(
                        P=po_state.blocks.P, A=po_state.blocks.A,
                        Q=po_state.blocks.Q, H=po_state.blocks.H,
                        τ=po_state.τ, e_vecs=po_state.e_vecs,
                        N=N, D=D
                    ),
                    data=(Y=po_state.Y_flat,),
                    options=(limit_stack_depth=1000,)
                )
            catch e
                @warn "PO inference failed at step $step, stopping early" exception=e
                done = true
                @goto next_step
            end

            acq = ucb_acquisition(res, po_state, cfg, N)
            k = select_next_point(acq.ucb, po_state.Y)
            if k == 0
                done = true
            else
                if cfg.animate
                    plt = plot_bo_step(step, k, po_state, acq, Ytrue, cfg)
                    push!(frames, plt)
                end

                y_new = eval_fn(row(Xo, k))
                po_state.Y[k] = (y_new .- po_state.μy) ./ po_state.σy

                # Fill Y_flat only where mask allows
                for d in 1:D
                    if po_state.mask[k, d]
                        po_state.Y_flat[(k-1)*D + d] = po_state.Y[k][d]
                    end
                end

                n_completed = step

                best = _current_best(po_state, cfg)
                n_obs = length(findall(!ismissing, po_state.Y))
                R_diag = 1.0 ./ po_state.τ

                push!(best_value_history, best.value)
                push!(n_observed_history, n_obs)
                push!(R_diag_history, R_diag)

                if step % cfg.log_every == 0
                    @info "PO Step $step/$(cfg.steps)" n_observed=n_obs best_value=round(best.value; digits=4)
                end
            end
            @label next_step
        end
        push!(step_times, t_step)
        done && break
    end

    best = _current_best(po_state, cfg)
    observed = findall(!ismissing, po_state.Y)
    R_learned = diagm(1.0 ./ po_state.τ)

    result = BOResult(best.index, best.value, best.y, observed, n_completed, R_learned,
        best_value_history, n_observed_history, R_diag_history,
        step_times, "state-space-po")
    (; result, frames)
end

# ─── Kernel-matrix baseline with partial observations ───────────────────────

"""
    BaselinePOState <: AbstractBOState

Mutable state for the LMC kernel-matrix GP baseline with partial observations.
"""
mutable struct BaselinePOState <: AbstractBOState
    W::Matrix{Float64}
    ℓs::Vector{Float64}
    σ2s::Vector{Float64}
    R_diag::Vector{Float64}
    dist_norm::Float64
    Xo::Matrix{Float64}
    mask::BitMatrix
    Y::Vector{Union{Missing, Vector{Float64}}}
    μy::Vector{Float64}
    σy::Vector{Float64}
    K_prior_full::Matrix{Float64}   # precomputed noise-free (D·N × D·N) LMC kernel
end

"""
    _lmc_full_kernel(Xo, W, ℓs, σ2s, dist_norm) -> Matrix{Float64}

Build the full noise-free (D·N × D·N) LMC covariance over all N points and all D
outputs, using point-major flat indexing pair (point i, output d) → (i-1)*D + d
(matching `Y_flat`).

    K[(i,d),(j,e)] = Σ_q W[d,q]·W[e,q]·matern32(dist(i,j)/dist_norm, ℓs[q], σ2s[q])

Computed once per `BaselinePOState`; the cov-restructuring baseline slices observed
sub-blocks out of this matrix each BO step rather than rebuilding from scratch.
"""
function _lmc_full_kernel(Xo::AbstractMatrix, W::Matrix{Float64},
                          ℓs::Vector{Float64}, σ2s::Vector{Float64},
                          dist_norm::Float64)
    N = size(Xo, 1)
    D = size(W, 1)
    Q = size(W, 2)

    dists = _pairwise_distances(Xo, Xo) ./ dist_norm

    K = zeros(D * N, D * N)
    for q in 1:Q
        Bq = W[:, q] * W[:, q]'   # D × D
        for j in 1:N, i in 1:N
            kval = _matern32_kernel(dists[i, j], ℓs[q], σ2s[q])
            for e in 1:D, d in 1:D
                K[(i-1)*D + d, (j-1)*D + e] += Bq[d, e] * kval
            end
        end
    end
    K
end

"""
    setup_baseline_po(cfg::ExperimentConfig, setup_data, mask::BitMatrix) -> BaselinePOState

Create a BaselinePOState from `setup_experiment` output and an observation mask.
"""
function setup_baseline_po(cfg::ExperimentConfig, setup_data, mask::BitMatrix)
    W_copy = copy(setup_data.W)
    Y_copy = copy(setup_data.Y)
    μy_copy = copy(setup_data.μy)
    σy_copy = copy(setup_data.σy)
    R_diag = fill(cfg.R_diag_init, cfg.D)

    K_prior_full = _lmc_full_kernel(setup_data.Xo, W_copy, copy(cfg.ℓs), copy(cfg.σ2s),
                                    setup_data.dist_norm)

    BaselinePOState(W_copy, copy(cfg.ℓs), copy(cfg.σ2s), R_diag,
                    setup_data.dist_norm, copy(setup_data.Xo),
                    mask, Y_copy, μy_copy, σy_copy, K_prior_full)
end

"""
    _lmc_predict_po(bl_state::BaselinePOState)

Predict at all N points using the cov-restructuring kernel-matrix GP baseline.

Rather than rebuilding a kernel from scratch, this restructures the precomputed
full LMC kernel `bl_state.K_prior_full` (D·N × D·N, noise-free): each step it
slices out the M×M sub-block over the currently-observed (point, output) pairs,
adds per-output noise, factorizes, and predicts. This is the per-step cost the
paper compares against state-space message passing.
"""
function _lmc_predict_po(bl_state::BaselinePOState)
    N = size(bl_state.Xo, 1)
    D = size(bl_state.W, 1)
    Kf = bl_state.K_prior_full

    # Restructure: flat indices (point-major) of the observed scalar entries
    obs_idx = Int[]
    obs_dim = Int[]
    for k in findall(!ismissing, bl_state.Y)
        for d in 1:D
            if bl_state.mask[k, d]
                push!(obs_idx, (k - 1) * D + d)
                push!(obs_dim, d)
            end
        end
    end
    # Observed sub-block + per-output noise on the diagonal
    K_obs = Kf[obs_idx, obs_idx] + Diagonal([bl_state.R_diag[d] for d in obs_dim])
    L = cholesky(Symmetric(K_obs) + 1e-8I)

    # Observation vector in obs_idx order
    y_obs = [bl_state.Y[(idx - 1) ÷ D + 1][obs_dim[a]] for (a, idx) in enumerate(obs_idx)]
    α = L \ y_obs

    # Vectorized prediction at all (point, output) entries
    cross = Kf[:, obs_idx]              # (D·N) × M, noise-free cross-covariance
    μ_flat = cross * α
    V = L.L \ cross'                    # M × (D·N)
    var_flat = diag(Kf) .- vec(sum(abs2, V; dims=1))

    μ_pred = Vector{Vector{Float64}}(undef, N)
    σ_pred = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        rng = (i - 1) * D + 1 : i * D
        μ_pred[i] = μ_flat[rng]
        σ_pred[i] = sqrt.(max.(var_flat[rng], 1e-10))
    end

    (; μ_pred, σ_pred)
end

"""
    baseline_po_ucb_acquisition(bl_state::BaselinePOState, cfg::ExperimentConfig)

Compute UCB acquisition using LMC kernel-matrix GP with partial observations.
"""
function baseline_po_ucb_acquisition(bl_state::BaselinePOState, cfg::ExperimentConfig)
    N = size(bl_state.Xo, 1)
    D = cfg.D
    s = cfg.s

    pred = _lmc_predict_po(bl_state)

    μ_pred = [pred.μ_pred[i] .* bl_state.σy .+ bl_state.μy for i in 1:N]
    σ_pred = [pred.σ_pred[i] .* bl_state.σy for i in 1:N]

    μs = [dot(s, μ_pred[i]) for i in 1:N]
    σs = [sqrt(sum((s[j] * σ_pred[i][j])^2 for j in 1:D)) for i in 1:N]
    ucb = μs .+ cfg.β .* σs

    (; ucb, μ_pred, σ_pred, μs, σs)
end

"""
    run_bo_baseline_po!(cfg::ExperimentConfig, eval_fn; bl_state, Ytrue) -> (; result, frames)

Run BO loop using the LMC kernel-matrix GP with partial observations.
"""
function run_bo_baseline_po!(cfg::ExperimentConfig, eval_fn; bl_state::BaselinePOState, Ytrue)
    N = cfg.N
    D = cfg.D
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
                acq = baseline_po_ucb_acquisition(bl_state, cfg)
            catch e
                @warn "LMC-PO prediction failed at step $step, stopping early" exception=e
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

                best = _current_best(bl_state, cfg)
                n_obs = length(findall(!ismissing, bl_state.Y))

                push!(best_value_history, best.value)
                push!(n_observed_history, n_obs)
                push!(R_diag_history, copy(bl_state.R_diag))

                if step % cfg.log_every == 0
                    @info "Baseline-PO Step $step/$(cfg.steps)" n_observed=n_obs best_value=round(best.value; digits=4)
                end
            end
            @label next_step
        end
        push!(step_times, t_step)
        done && break
    end

    best = _current_best(bl_state, cfg)
    observed = findall(!ismissing, bl_state.Y)
    R_learned = diagm(bl_state.R_diag)

    result = BOResult(best.index, best.value, best.y, observed, n_completed, R_learned,
                      best_value_history, n_observed_history, R_diag_history,
                      step_times, "kernel-matrix-po-restructure")
    (; result, frames)
end
