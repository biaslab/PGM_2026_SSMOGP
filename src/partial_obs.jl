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
    rmse_history = Float64[]
    mnll_history = Float64[]
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

                # Held-out regression metrics
                rmse = _compute_rmse_test(acq.μ_pred, Ytrue, po_state.Y, D)
                mnll = _compute_mnll_test(acq.μ_pred, acq.σ_pred, Ytrue, po_state.Y, D)
                push!(rmse_history, rmse)
                push!(mnll_history, mnll)

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
        rmse_history, mnll_history,
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

    BaselinePOState(W_copy, copy(cfg.ℓs), copy(cfg.σ2s), R_diag,
                    setup_data.dist_norm, copy(setup_data.Xo),
                    mask, Y_copy, μy_copy, σy_copy)
end

"""
    _lmc_predict_po(bl_state::BaselinePOState)

Predict at all N points using a kernel-matrix GP that handles partial observations.

Builds an M×M kernel matrix where M = number of observed (point, output) pairs.
Each entry: K[a,b] = Σ_q W[d_a,q]*W[d_b,q]*k_q(x_ia, x_jb) + R[d_a]*δ(ia==jb, d_a==d_b)
"""
function _lmc_predict_po(bl_state::BaselinePOState)
    N = size(bl_state.Xo, 1)
    D = size(bl_state.W, 1)
    Q = size(bl_state.W, 2)

    observed_pts = findall(!ismissing, bl_state.Y)
    n_obs = length(observed_pts)

    # Build list of (point_index, output_dim) pairs that are observed
    obs_pairs = Tuple{Int, Int}[]
    for k in observed_pts
        for d in 1:D
            if bl_state.mask[k, d]
                push!(obs_pairs, (k, d))
            end
        end
    end
    M = length(obs_pairs)

    # Pairwise distances (normalized)
    dists_full = _pairwise_distances(bl_state.Xo, bl_state.Xo) ./ bl_state.dist_norm

    # Build M×M kernel matrix
    K = zeros(M, M)
    for b in 1:M, a in 1:M
        ia, da = obs_pairs[a]
        ib, db = obs_pairs[b]
        kval = 0.0
        for q in 1:Q
            kval += bl_state.W[da, q] * bl_state.W[db, q] *
                    _matern32_kernel(dists_full[ia, ib], bl_state.ℓs[q], bl_state.σ2s[q])
        end
        if ia == ib && da == db
            kval += bl_state.R_diag[da]
        end
        K[a, b] = kval
    end

    K_jit = Symmetric(K) + 1e-8I
    L = cholesky(K_jit)

    # Observation vector (matching obs_pairs order)
    y_obs = zeros(M)
    for (idx, (k, d)) in enumerate(obs_pairs)
        y_obs[idx] = bl_state.Y[k][d]
    end
    α = L \ y_obs

    # Prior self-variance
    prior_var = _lmc_self_variance(bl_state.W, bl_state.σ2s)

    # Predict at each candidate point
    μ_pred = Vector{Vector{Float64}}(undef, N)
    σ_pred = Vector{Vector{Float64}}(undef, N)

    for i in 1:N
        μ_d = zeros(D)
        var_d = copy(prior_var)

        for d in 1:D
            # Cross-covariance between (i, d) and all observed pairs
            kx = zeros(M)
            for (idx, (k, dk)) in enumerate(obs_pairs)
                for q in 1:Q
                    kx[idx] += bl_state.W[d, q] * bl_state.W[dk, q] *
                               _matern32_kernel(dists_full[i, k], bl_state.ℓs[q], bl_state.σ2s[q])
                end
            end

            μ_d[d] = dot(kx, α)
            v = L.L \ kx
            var_d[d] -= dot(v, v)
            var_d[d] = max(var_d[d], 1e-10)
        end

        μ_pred[i] = μ_d
        σ_pred[i] = sqrt.(var_d)
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
    rmse_history = Float64[]
    mnll_history = Float64[]
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

                # Held-out regression metrics
                rmse = _compute_rmse_test(acq.μ_pred, Ytrue, bl_state.Y, D)
                mnll = _compute_mnll_test(acq.μ_pred, acq.σ_pred, Ytrue, bl_state.Y, D)
                push!(rmse_history, rmse)
                push!(mnll_history, mnll)

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
                      rmse_history, mnll_history,
                      step_times, "kernel-matrix-po")
    (; result, frames)
end

# ─── Random baseline ───────────────────────────────────────────────────────

"""
    run_bo_random!(cfg, eval_fn; Y, Xo, μy, σy, s, rng) -> (; best_value_history, step_times)

Run BO loop with random (no surrogate) point selection.
At each step, randomly selects an unobserved candidate and evaluates it.
Tracks only best scalarized value history and step times.
"""
function run_bo_random!(cfg::ExperimentConfig, eval_fn; Y, Xo, μy, σy, rng)
    best_value_history = Float64[]
    step_times = Float64[]
    best_val = -Inf

    # Check if any seed observations already exist
    for i in eachindex(Y)
        if !ismissing(Y[i])
            y_orig = Y[i] .* σy .+ μy
            val = dot(cfg.s, y_orig)
            best_val = max(best_val, val)
        end
    end

    for step in 1:cfg.steps
        t_step = @elapsed begin
            unobs = findall(ismissing, Y)
            if isempty(unobs)
                break
            end
            k = rand(rng, unobs)
            y_new = eval_fn(row(Xo, k))
            Y[k] = (y_new .- μy) ./ σy
            val = dot(cfg.s, y_new)
            best_val = max(best_val, val)
        end
        push!(best_value_history, best_val)
        push!(step_times, t_step)
    end

    (; best_value_history, step_times)
end

# ─── Comparison runner ──────────────────────────────────────────────────────

"""
    run_po_comparison(cfg_template::ExperimentConfig, eval_fn; seeds, output_dir) -> results

Run 4-way comparison: SS-full, SS-PO, KM-full, KM-PO across multiple seeds.

Demonstrates that the FFG/SS-GP handles partial observations naturally while
the kernel-matrix baseline requires restructuring its covariance matrix.
"""
function run_po_comparison(cfg_template::ExperimentConfig, eval_fn; seeds=0:4, output_dir="data/partial_obs")
    mkpath(output_dir)
    all_results = Dict{String, Any}[]

    # Build partial-obs config
    cfg_po = ExperimentConfig(;
        N=cfg_template.N, d=cfg_template.d, Q=cfg_template.Q, D=cfg_template.D,
        ℓs=cfg_template.ℓs, σ2s=cfg_template.σ2s, β=cfg_template.β, s=cfg_template.s,
        n_seed=cfg_template.n_seed, steps=cfg_template.steps,
        tune_every=cfg_template.tune_every, R_diag_init=cfg_template.R_diag_init,
        animate=false, log_every=cfg_template.log_every, seed=0,
        obs_pattern=:sensor_groups, obs_frac=cfg_template.obs_frac
    )

    for seed in seeds
        @info "=== Partial-obs comparison seed=$seed ==="
        cfg_full = ExperimentConfig(;
            N=cfg_template.N, d=cfg_template.d, Q=cfg_template.Q, D=cfg_template.D,
            ℓs=cfg_template.ℓs, σ2s=cfg_template.σ2s, β=cfg_template.β, s=cfg_template.s,
            n_seed=cfg_template.n_seed, steps=cfg_template.steps,
            tune_every=cfg_template.tune_every, R_diag_init=cfg_template.R_diag_init,
            animate=false, log_every=cfg_template.log_every, seed=seed,
            obs_pattern=:full
        )
        cfg_partial = ExperimentConfig(;
            N=cfg_po.N, d=cfg_po.d, Q=cfg_po.Q, D=cfg_po.D,
            ℓs=cfg_po.ℓs, σ2s=cfg_po.σ2s, β=cfg_po.β, s=cfg_po.s,
            n_seed=cfg_po.n_seed, steps=cfg_po.steps,
            tune_every=cfg_po.tune_every, R_diag_init=cfg_po.R_diag_init,
            animate=false, log_every=cfg_po.log_every, seed=seed,
            obs_pattern=:sensor_groups
        )

        # ── SS-GP full observations ──
        @info "Running SS-GP full (seed=$seed)"
        setup_data = setup_experiment(cfg_full, eval_fn)
        po_full = setup_po(cfg_full, setup_data)
        out_ss_full = run_bo_po!(cfg_full, eval_fn;
            po_state=po_full, Xo=setup_data.Xo, Δ=setup_data.Δ, Ytrue=setup_data.Ytrue)

        # ── SS-GP partial observations ──
        @info "Running SS-GP partial (seed=$seed)"
        setup_data2 = setup_experiment(cfg_partial, eval_fn)
        po_partial = setup_po(cfg_partial, setup_data2)
        out_ss_po = run_bo_po!(cfg_partial, eval_fn;
            po_state=po_partial, Xo=setup_data2.Xo, Δ=setup_data2.Δ, Ytrue=setup_data2.Ytrue)

        # ── KM-GP full observations ──
        @info "Running KM-GP full (seed=$seed)"
        setup_data3 = setup_experiment(cfg_full, eval_fn)
        mask_full = _generate_obs_mask(cfg_full, cfg_full.N, MersenneTwister(seed + 1000))
        bl_full = setup_baseline_po(cfg_full, setup_data3, mask_full)
        out_km_full = run_bo_baseline_po!(cfg_full, eval_fn;
            bl_state=bl_full, Ytrue=setup_data3.Ytrue)

        # ── KM-GP partial observations ──
        @info "Running KM-GP partial (seed=$seed)"
        setup_data4 = setup_experiment(cfg_partial, eval_fn)
        mask_partial = _generate_obs_mask(cfg_partial, cfg_partial.N, MersenneTwister(seed + 1000))
        bl_partial = setup_baseline_po(cfg_partial, setup_data4, mask_partial)
        out_km_po = run_bo_baseline_po!(cfg_partial, eval_fn;
            bl_state=bl_partial, Ytrue=setup_data4.Ytrue)

        # ── Random baseline ──
        @info "Running Random baseline (seed=$seed)"
        setup_data5 = setup_experiment(cfg_full, eval_fn)
        Y_rand = copy(setup_data5.Y)
        out_random = run_bo_random!(cfg_full, eval_fn;
            Y=Y_rand, Xo=setup_data5.Xo,
            μy=setup_data5.μy, σy=setup_data5.σy,
            rng=MersenneTwister(seed + 2000))

        _bo_summary(r) = Dict(
            "best_value_history" => r.best_value_history,
            "step_times"         => r.step_times,
            "best_value"         => r.best_value,
            "n_iterations"       => r.n_iterations,
            "total_time"         => sum(r.step_times),
            "rmse_history"       => r.rmse_history,
            "mnll_history"       => r.mnll_history,
        )

        push!(all_results, Dict(
            "seed" => seed,
            "ss_full" => _bo_summary(out_ss_full.result),
            "ss_po"   => _bo_summary(out_ss_po.result),
            "km_full" => _bo_summary(out_km_full.result),
            "km_po"   => _bo_summary(out_km_po.result),
            "random"  => Dict(
                "best_value_history" => out_random.best_value_history,
                "step_times"         => out_random.step_times,
                "best_value"         => isempty(out_random.best_value_history) ? NaN : out_random.best_value_history[end],
                "total_time"         => sum(out_random.step_times),
            ),
        ))
    end

    # Save JSON — allow NaN so transient RxInfer numerical blowups don't kill the write.
    json_path = joinpath(output_dir, "comparison.json")
    open(json_path, "w") do io
        JSON.json(io, all_results; pretty=2, allownan=true)
    end
    @info "Saved partial-obs comparison to $json_path"

    _plot_po_comparison(all_results, output_dir)

    all_results
end

"""
    _plot_po_comparison(results, output_dir)

Generate convergence, timing, and regression metric plots for the partial-obs experiment.
"""
function _plot_po_comparison(results, output_dir)
    methods = ["ss_full", "ss_po", "km_full", "km_po"]
    labels = ["SS-GP (full)", "SS-GP (partial)", "KM-GP (full)", "KM-GP (partial)"]
    colors = [:blue, :cyan, :red, :orange]

    max_steps = maximum(r -> maximum(length(r[m]["best_value_history"]) for m in methods), results)

    function _pad(v, n)
        length(v) >= n && return v[1:n]
        vcat(v, fill(NaN, n - length(v)))
    end

    _nanmean(x) = mean(filter(!isnan, x))
    _nanstd(x)  = (v = filter(!isnan, x); length(v) > 1 ? std(v) : 0.0)
    _nanmedian(x) = (v = filter(!isnan, x); isempty(v) ? NaN : median(v))

    steps = 1:max_steps

    linestyles = [:solid, :dash, :solid, :dash]
    theme_kw = publication_theme_kwargs()

    # Plot 1: Convergence (with random baseline)
    save_plot(joinpath(output_dir, "convergence")) do
        p = plot(; xlabel="BO Step", ylabel="Best Scalarized Value",
                 legend=:bottomright, theme_kw...)
        for (m, lab, col, ls) in zip(methods, labels, colors, linestyles)
            histories = hcat([_pad(r[m]["best_value_history"], max_steps) for r in results]...)
            m_mean = [_nanmean(histories[i, :]) for i in 1:max_steps]
            m_std  = [_nanstd(histories[i, :])  for i in 1:max_steps]
            plot!(p, steps, m_mean, ribbon=m_std, fillalpha=0.15, lw=2,
                  label=lab, color=col, linestyle=ls)
        end
        # Random baseline
        if haskey(first(results), "random")
            rand_histories = hcat([_pad(r["random"]["best_value_history"], max_steps) for r in results]...)
            r_mean = [_nanmean(rand_histories[i, :]) for i in 1:max_steps]
            r_std  = [_nanstd(rand_histories[i, :])  for i in 1:max_steps]
            plot!(p, steps, r_mean, ribbon=r_std, fillalpha=0.1, lw=2,
                  label="Random", color=:gray, linestyle=:dot)
        end
        p
    end

    # Plot 2: Timing per step
    save_plot(joinpath(output_dir, "timing")) do
        p = plot(; xlabel="BO Step", ylabel="Median Time per Step (s)", yscale=:log10,
                 legend=:topright, theme_kw...)
        for (m, lab, col, ls) in zip(methods, labels, colors, linestyles)
            times = hcat([_pad(r[m]["step_times"], max_steps) for r in results]...)
            t_med = [_nanmedian(times[i, :]) for i in 1:max_steps]
            plot!(p, steps, t_med, lw=2, label=lab, color=col, linestyle=ls)
        end
        p
    end

    # Plot 3: RMSE on held-out points (median + IQR, robust to transient instabilities)
    # Pre-filter finite-but-extreme numerical artifacts from RxInfer's Kalman smoother.
    # Normal RMSE is ~0.3 and normal |MNLL| is ~0.5; anything above `thr` is an artifact.
    _clean(x, thr) = filter(v -> !isnan(v) && isfinite(v) && abs(v) <= thr, x)
    _medf(x, thr)  = (v = _clean(x, thr); isempty(v) ? NaN : median(v))
    _q25f(x, thr)  = (v = _clean(x, thr); length(v) >= 2 ? quantile(v, 0.25) : (isempty(v) ? NaN : v[1]))
    _q75f(x, thr)  = (v = _clean(x, thr); length(v) >= 2 ? quantile(v, 0.75) : (isempty(v) ? NaN : v[1]))

    if haskey(first(results)["ss_full"], "rmse_history")
        save_plot(joinpath(output_dir, "rmse")) do
            thr = 10.0
            p = plot(; xlabel="BO Step", ylabel="RMSE (held-out)",
                     legend=:topright, theme_kw...)
            for (m, lab, col, ls) in zip(methods, labels, colors, linestyles)
                histories = hcat([_pad(r[m]["rmse_history"], max_steps) for r in results]...)
                m_med = [_medf(histories[i, :], thr) for i in 1:max_steps]
                m_lo  = m_med .- [_q25f(histories[i, :], thr) for i in 1:max_steps]
                m_hi  = [_q75f(histories[i, :], thr) for i in 1:max_steps] .- m_med
                plot!(p, steps, m_med, ribbon=(m_lo, m_hi), fillalpha=0.15, lw=2,
                      label=lab, color=col, linestyle=ls)
            end
            p
        end
    end

    # Plot 4: MNLL on held-out points (median + IQR)
    if haskey(first(results)["ss_full"], "mnll_history")
        save_plot(joinpath(output_dir, "mnll")) do
            thr = 10.0
            p = plot(; xlabel="BO Step", ylabel="MNLL (held-out)",
                     legend=:topright, theme_kw...)
            for (m, lab, col, ls) in zip(methods, labels, colors, linestyles)
                histories = hcat([_pad(r[m]["mnll_history"], max_steps) for r in results]...)
                m_med = [_medf(histories[i, :], thr) for i in 1:max_steps]
                m_lo  = m_med .- [_q25f(histories[i, :], thr) for i in 1:max_steps]
                m_hi  = [_q75f(histories[i, :], thr) for i in 1:max_steps] .- m_med
                plot!(p, steps, m_med, ribbon=(m_lo, m_hi), fillalpha=0.15, lw=2,
                      label=lab, color=col, linestyle=ls)
            end
            p
        end
    end
end
