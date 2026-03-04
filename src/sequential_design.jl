# Sequential experimental design: Bayesian Quadrature and Active Learning
#
# Both BQ and AL use the same GP surrogate and loop structure as BO — only the
# acquisition function (max total predictive variance) and tracked metrics differ.
# This file provides a unified comparison runner that tracks both metrics simultaneously.

"""
    baseline_po_variance_acquisition(bl_state, cfg) -> (; score, μ_pred, σ_pred)

Compute total predictive variance acquisition using LMC kernel-matrix GP with partial observations.
"""
function baseline_po_variance_acquisition(bl_state::BaselinePOState, cfg::ExperimentConfig)
    N = size(bl_state.Xo, 1)

    pred = _lmc_predict_po(bl_state)

    μ_pred = [pred.μ_pred[i] .* bl_state.σy .+ bl_state.μy for i in 1:N]
    σ_pred = [pred.σ_pred[i] .* bl_state.σy for i in 1:N]

    score = [sum(σ_pred[i] .^ 2) for i in 1:N]
    (; score, μ_pred, σ_pred)
end

"""
    _compute_integral_error(μ_pred, Ytrue, N, D) -> Float64

Compute integral error for Bayesian quadrature.

Estimates the integral as the mean of predictive means across all candidate points,
and compares to the ground-truth integral (mean of true function values).

    Z_hat[d] = (1/N) Σᵢ μ_pred[i][d]
    Z_true[d] = (1/N) Σᵢ Ytrue[i][d]
    integral_error = ‖Z_hat - Z_true‖₂
"""
function _compute_integral_error(μ_pred::Vector{Vector{Float64}}, Ytrue::Vector{Vector{Float64}}, N::Int, D::Int)
    Z_hat = zeros(D)
    Z_true = zeros(D)
    for i in 1:N
        Z_hat .+= μ_pred[i]
        Z_true .+= Ytrue[i]
    end
    Z_hat ./= N
    Z_true ./= N
    sqrt(sum((Z_hat .- Z_true) .^ 2))
end

"""
    _compute_rmse(μ_pred, Ytrue, N, D) -> Float64

Compute root mean squared error for active learning.

    RMSE = sqrt(1/(N*D) Σᵢ Σ_d (μ_pred[i][d] - Ytrue[i][d])²)
"""
function _compute_rmse(μ_pred::Vector{Vector{Float64}}, Ytrue::Vector{Vector{Float64}}, N::Int, D::Int)
    mse = 0.0
    for i in 1:N
        for d in 1:D
            mse += (μ_pred[i][d] - Ytrue[i][d])^2
        end
    end
    sqrt(mse / (N * D))
end

"""
    _compute_mnll(μ_pred, σ_pred, Ytrue, Y, D) -> Float64

Compute mean negative log-likelihood (negative log predictive density) over unobserved points.

Only evaluates on test points (where `ismissing(Y[i])`) to avoid degenerate log-likelihoods
at observed points where the GP predictive variance is near zero.

    MNLL = 1/(N_test*D) Σᵢ Σ_d [ ½ log(2π σ²ᵢd) + (yᵢd - μᵢd)² / (2 σ²ᵢd) ]
"""
function _compute_mnll(μ_pred::Vector{Vector{Float64}}, σ_pred::Vector{Vector{Float64}},
                       Ytrue::Vector{Vector{Float64}}, Y::Vector{Union{Missing, Vector{Float64}}}, D::Int)
    nll = 0.0
    n_test = 0
    for i in eachindex(Y)
        ismissing(Y[i]) || continue
        n_test += 1
        for d in 1:D
            σ2 = max(σ_pred[i][d]^2, 1e-12)
            nll += 0.5 * log(2π * σ2) + (Ytrue[i][d] - μ_pred[i][d])^2 / (2σ2)
        end
    end
    n_test == 0 ? NaN : nll / (n_test * D)
end

"""
    run_sd_po!(cfg, eval_fn; po_state, Xo, Δ, Ytrue) -> (; result)

Run sequential design loop with the partial-observation state-space GP model.

Uses max total predictive variance as the acquisition function and tracks
both integral error (BQ) and RMSE (AL) metrics.
"""
function run_sd_po!(cfg::ExperimentConfig, eval_fn; po_state::POState, Xo, Δ, Ytrue)
    N = cfg.N
    D = cfg.D

    integral_error_history = Float64[]
    rmse_history = Float64[]
    mnll_history = Float64[]
    n_observed_history = Int[]
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
                @warn "SD-PO inference failed at step $step, stopping early" exception=e
                done = true
                @goto next_step
            end

            acq = variance_acquisition(res, po_state, cfg, N)

            if any(!isfinite, acq.score)
                @warn "SD-PO posteriors diverged at step $step, stopping early"
                done = true
                @goto next_step
            end

            k = select_next_point(acq.score, po_state.Y)
            if k == 0
                done = true
            else
                y_new = eval_fn(row(Xo, k))
                po_state.Y[k] = (y_new .- po_state.μy) ./ po_state.σy

                for d in 1:D
                    if po_state.mask[k, d]
                        po_state.Y_flat[(k-1)*D + d] = po_state.Y[k][d]
                    end
                end

                n_completed = step

                ie = _compute_integral_error(acq.μ_pred, Ytrue, N, D)
                rmse = _compute_rmse(acq.μ_pred, Ytrue, N, D)
                mnll = _compute_mnll(acq.μ_pred, acq.σ_pred, Ytrue, po_state.Y, D)

                n_obs = length(findall(!ismissing, po_state.Y))

                push!(integral_error_history, ie)
                push!(rmse_history, rmse)
                push!(mnll_history, mnll)
                push!(n_observed_history, n_obs)

                if step % cfg.log_every == 0
                    @info "SD-PO Step $step/$(cfg.steps)" n_observed=n_obs ie=round(ie; digits=6) rmse=round(rmse; digits=6) mnll=round(mnll; digits=6)
                end
            end
            @label next_step
        end
        push!(step_times, t_step)
        done && break
    end

    observed = findall(!ismissing, po_state.Y)
    result = SDResult(observed, n_completed, integral_error_history, rmse_history,
                      mnll_history, n_observed_history, step_times, "state-space-po")
    (; result)
end

"""
    run_sd_baseline_po!(cfg, eval_fn; bl_state, Ytrue) -> (; result)

Run sequential design loop using the LMC kernel-matrix GP with partial observations.

Uses max total predictive variance as the acquisition function and tracks
both integral error (BQ) and RMSE (AL) metrics.
"""
function run_sd_baseline_po!(cfg::ExperimentConfig, eval_fn; bl_state::BaselinePOState, Ytrue)
    N = cfg.N
    D = cfg.D
    Xo = bl_state.Xo

    integral_error_history = Float64[]
    rmse_history = Float64[]
    mnll_history = Float64[]
    n_observed_history = Int[]
    step_times = Float64[]

    n_completed = 0
    done = false
    for step in 1:cfg.steps
        t_step = @elapsed begin
            local acq
            try
                acq = baseline_po_variance_acquisition(bl_state, cfg)
            catch e
                @warn "SD-KM-PO prediction failed at step $step, stopping early" exception=e
                done = true
                @goto next_step
            end

            if any(!isfinite, acq.score)
                @warn "SD-KM-PO posteriors diverged at step $step, stopping early"
                done = true
                @goto next_step
            end

            k = select_next_point(acq.score, bl_state.Y)
            if k == 0
                done = true
            else
                y_new = eval_fn(row(Xo, k))
                bl_state.Y[k] = (y_new .- bl_state.μy) ./ bl_state.σy

                n_completed = step

                ie = _compute_integral_error(acq.μ_pred, Ytrue, N, D)
                rmse = _compute_rmse(acq.μ_pred, Ytrue, N, D)
                mnll = _compute_mnll(acq.μ_pred, acq.σ_pred, Ytrue, bl_state.Y, D)

                n_obs = length(findall(!ismissing, bl_state.Y))

                push!(integral_error_history, ie)
                push!(rmse_history, rmse)
                push!(mnll_history, mnll)
                push!(n_observed_history, n_obs)

                if step % cfg.log_every == 0
                    @info "SD-KM-PO Step $step/$(cfg.steps)" n_observed=n_obs ie=round(ie; digits=6) rmse=round(rmse; digits=6) mnll=round(mnll; digits=6)
                end
            end
            @label next_step
        end
        push!(step_times, t_step)
        done && break
    end

    observed = findall(!ismissing, bl_state.Y)
    result = SDResult(observed, n_completed, integral_error_history, rmse_history,
                      mnll_history, n_observed_history, step_times, "kernel-matrix-po")
    (; result)
end

"""
    run_sd_comparison(cfg_template, eval_fn; seeds, output_dir) -> results

Run sequential design comparison across multiple seeds.

When `run_partial=true` (default), runs 4-way: SS-full, SS-PO, KM-full, KM-PO.
When `run_partial=false`, runs 2-way: SS-full, KM-full only.

Tracks both integral error and RMSE simultaneously (same GP posteriors).
Saves `comparison.json` with all per-seed results.
"""
function run_sd_comparison(cfg_template::ExperimentConfig, eval_fn; seeds=0:4, output_dir="data/sd", run_partial::Bool=true)
    mkpath(output_dir)
    all_results = Dict{String, Any}[]

    cfg_po = if run_partial
        ExperimentConfig(;
            N=cfg_template.N, d=cfg_template.d, Q=cfg_template.Q, D=cfg_template.D,
            ℓs=cfg_template.ℓs, σ2s=cfg_template.σ2s, β=cfg_template.β, s=cfg_template.s,
            n_seed=cfg_template.n_seed, steps=cfg_template.steps,
            tune_every=cfg_template.tune_every, R_diag_init=cfg_template.R_diag_init,
            animate=false, log_every=cfg_template.log_every, seed=0,
            obs_pattern=:sensor_groups, obs_frac=cfg_template.obs_frac
        )
    else
        nothing
    end

    for seed in seeds
        @info "=== SD comparison seed=$seed ==="
        cfg_full = ExperimentConfig(;
            N=cfg_template.N, d=cfg_template.d, Q=cfg_template.Q, D=cfg_template.D,
            ℓs=cfg_template.ℓs, σ2s=cfg_template.σ2s, β=cfg_template.β, s=cfg_template.s,
            n_seed=cfg_template.n_seed, steps=cfg_template.steps,
            tune_every=cfg_template.tune_every, R_diag_init=cfg_template.R_diag_init,
            animate=false, log_every=cfg_template.log_every, seed=seed,
            obs_pattern=:full
        )
        cfg_partial = if run_partial
            ExperimentConfig(;
                N=cfg_po.N, d=cfg_po.d, Q=cfg_po.Q, D=cfg_po.D,
                ℓs=cfg_po.ℓs, σ2s=cfg_po.σ2s, β=cfg_po.β, s=cfg_po.s,
                n_seed=cfg_po.n_seed, steps=cfg_po.steps,
                tune_every=cfg_po.tune_every, R_diag_init=cfg_po.R_diag_init,
                animate=false, log_every=cfg_po.log_every, seed=seed,
                obs_pattern=:sensor_groups
            )
        else
            nothing
        end

        # ── SS-GP full observations ──
        @info "Running SS-GP full (seed=$seed)"
        setup_data = setup_experiment(cfg_full, eval_fn)
        po_full = setup_po(cfg_full, setup_data)
        out_ss_full = run_sd_po!(cfg_full, eval_fn;
            po_state=po_full, Xo=setup_data.Xo, Δ=setup_data.Δ, Ytrue=setup_data.Ytrue)

        # ── SS-GP partial observations ──
        out_ss_po = if run_partial
            @info "Running SS-GP partial (seed=$seed)"
            setup_data2 = setup_experiment(cfg_partial, eval_fn)
            po_partial = setup_po(cfg_partial, setup_data2)
            run_sd_po!(cfg_partial, eval_fn;
                po_state=po_partial, Xo=setup_data2.Xo, Δ=setup_data2.Δ, Ytrue=setup_data2.Ytrue)
        else
            nothing
        end

        # ── KM-GP full observations ──
        @info "Running KM-GP full (seed=$seed)"
        setup_data3 = setup_experiment(cfg_full, eval_fn)
        mask_full = _generate_obs_mask(cfg_full, cfg_full.N, MersenneTwister(seed + 1000))
        bl_full = setup_baseline_po(cfg_full, setup_data3, mask_full)
        out_km_full = run_sd_baseline_po!(cfg_full, eval_fn;
            bl_state=bl_full, Ytrue=setup_data3.Ytrue)

        # ── KM-GP partial observations ──
        out_km_po = if run_partial
            @info "Running KM-GP partial (seed=$seed)"
            setup_data4 = setup_experiment(cfg_partial, eval_fn)
            mask_partial = _generate_obs_mask(cfg_partial, cfg_partial.N, MersenneTwister(seed + 1000))
            bl_partial = setup_baseline_po(cfg_partial, setup_data4, mask_partial)
            run_sd_baseline_po!(cfg_partial, eval_fn;
                bl_state=bl_partial, Ytrue=setup_data4.Ytrue)
        else
            nothing
        end

        _final(v) = isempty(v) ? nothing : v[end]
        _summary(r) = Dict(
            "ie_final"     => _final(r.integral_error_history),
            "rmse_final"   => _final(r.rmse_history),
            "mnll_final"   => _final(r.mnll_history),
            "n_iterations" => r.n_iterations,
            "step_times"   => r.step_times,
            "total_time"   => sum(r.step_times),
        )

        seed_result = Dict{String, Any}(
            "seed"    => seed,
            "ss_full" => _summary(out_ss_full.result),
            "km_full" => _summary(out_km_full.result),
        )
        if run_partial
            seed_result["ss_po"] = _summary(out_ss_po.result)
            seed_result["km_po"] = _summary(out_km_po.result)
        end
        push!(all_results, seed_result)
    end

    # Save JSON
    json_path = joinpath(output_dir, "comparison.json")
    open(json_path, "w") do io
        JSON.print(io, all_results, 2)
    end
    @info "Saved SD comparison to $json_path"

    all_results
end

"""
    _plot_sd_timing(results, output_dir)

Generate timing comparison plot for a sequential design experiment.
"""
function _plot_sd_timing(results, output_dir)
    has_partial = haskey(first(results), "ss_po")
    methods = has_partial ? ["ss_full", "ss_po", "km_full", "km_po"] : ["ss_full", "km_full"]
    labels = has_partial ? ["SS-GP (full)", "SS-GP (partial)", "KM-GP (full)", "KM-GP (partial)"] : ["SS-GP", "KM-GP"]
    colors = has_partial ? [:blue, :cyan, :red, :orange] : [:blue, :red]
    linestyles = has_partial ? [:solid, :dash, :solid, :dash] : [:solid, :solid]

    max_steps = maximum(r -> maximum(length(r[m]["step_times"]) for m in methods), results)

    function _pad(v, n)
        length(v) >= n && return v[1:n]
        vcat(v, fill(NaN, n - length(v)))
    end

    _nanmedian(x) = (v = filter(!isnan, x); isempty(v) ? NaN : median(v))

    steps = 1:max_steps
    theme_kw = publication_theme_kwargs()

    save_plot(joinpath(output_dir, "timing")) do
        p = plot(; xlabel="Step", ylabel="Median Time per Step (s)", yscale=:log10,
                 legend=:topright, theme_kw...)
        for (m, lab, col, ls) in zip(methods, labels, colors, linestyles)
            times = hcat([_pad(r[m]["step_times"], max_steps) for r in results]...)
            t_med = [_nanmedian(times[i, :]) for i in 1:max_steps]
            plot!(p, steps, t_med, lw=2, label=lab, color=col, linestyle=ls)
        end
        p
    end
    save_plot(joinpath(output_dir, "timelimit")) do
        p = plot(; xlabel="Time (s)", ylabel="Steps", #yscale=:log10,
                legend=:bottomright, theme_kw...)
        for (m, lab, col, ls) in zip(methods, labels, colors, linestyles)
            times = hcat([_pad(r[m]["step_times"], max_steps) for r in results]...)
            cum_times = cumsum(times, dims=1)
            t_cum = [_nanmedian(cum_times[i, :]) for i in 1:max_steps]
            plot!(p, t_cum, steps, lw=2, label=lab, color=col, linestyle=ls)
        end
        p
    end
end

"""
    run_bq_comparison(cfg, eval_fn; seeds, output_dir) -> results

Run Bayesian quadrature comparison: calls `run_sd_comparison` and generates timing plot.

Runs SS-full vs KM-full only (no partial observations).
"""
function run_bq_comparison(cfg::ExperimentConfig, eval_fn; seeds=0:4, output_dir="data/quadrature")
    results = run_sd_comparison(cfg, eval_fn; seeds=seeds, output_dir=output_dir, run_partial=false)
    _plot_sd_timing(results, output_dir)
    results
end