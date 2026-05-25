# Input-dimension sweep on the synthetic sensor-network benchmark.
#
# Stresses the NN-chain ordering by varying `d` and comparing three methods:
#   - SS-GP   (uses NN chain — sensitive to d)
#   - KM-GP   (full kernel matrix — d-agnostic, "oracle" baseline)
#   - Random  (uniform unobserved-point picker — lower bound)
#
# For each (d, seed) the runner also records `nn_chain_quality(Xo)` so we can
# correlate any BO regret gap with raw chain stretch.

"""
    _run_random_bo(cfg, eval_fn; Xo, setup_data) -> NamedTuple

Random-acquisition baseline. At each step picks a uniformly random unobserved
chain index, evaluates `eval_fn`, and tracks the running best scalarized value.
Uses `cfg.seed + 9999` so its draws are deterministic and independent of the
GP-method seeds.
"""
function _run_random_bo(cfg::ExperimentConfig, eval_fn; Xo::AbstractMatrix,
                        setup_data)
    Y     = copy(setup_data.Y)
    μy    = copy(setup_data.μy)
    σy    = copy(setup_data.σy)
    rng   = MersenneTwister(cfg.seed + 9999)

    best_value_history = Float64[]
    step_times         = Float64[]
    n_completed        = 0

    for step in 1:cfg.steps
        t_step = @elapsed begin
            unobserved = findall(ismissing, Y)
            if isempty(unobserved)
                break
            end
            k = unobserved[rand(rng, 1:length(unobserved))]
            y_new = eval_fn(row(Xo, k))
            Y[k] = (y_new .- μy) ./ σy

            best_val = -Inf
            for kk in findall(!ismissing, Y)
                y_orig = Y[kk] .* σy .+ μy
                v = dot(cfg.s, y_orig)
                if v > best_val
                    best_val = v
                end
            end
            push!(best_value_history, best_val)
            n_completed = step
        end
        push!(step_times, t_step)
    end

    best_value = isempty(best_value_history) ? -Inf : last(best_value_history)
    (; best_value_history, step_times, best_value, n_iterations=n_completed)
end

"""
    run_dim_sweep(cfg_template, eval_fn_factory; ds, seeds, output_dir) -> Vector{Dict}

Sweep BO performance over input dimensions `ds`, comparing SS-GP, KM-GP, and a
random-acquisition baseline. `eval_fn_factory(d)` must return a function whose
input is a length-`d` vector. The other config fields (`D`, `Q`, `N`, etc.) are
held fixed.

For each (d, seed):
  1. Build a config with this `d` and `seed`.
  2. Call `setup_experiment` once and reuse `setup_data` for all methods.
  3. Run SS-GP via `run_bo_po!` (full mask) and KM-GP via `run_bo_baseline_po!`.
  4. Run the random baseline.
  5. Record `nn_chain_quality(Xo)`.

Writes `comparison.json` and rendered figures to `output_dir`.
"""
function run_dim_sweep(cfg_template::ExperimentConfig, eval_fn_factory;
                      ds::Vector{Int}=[2, 4, 8, 16, 32], seeds=0:4,
                      output_dir::String="data/dim_sweep")
    mkpath(output_dir)
    all_results = Dict{String, Any}[]

    for d in ds
        for seed in seeds
            @info "=== dim_sweep d=$d seed=$seed ==="
            cfg = ExperimentConfig(;
                N=cfg_template.N, d=d, Q=cfg_template.Q, D=cfg_template.D,
                ℓs=cfg_template.ℓs, σ2s=cfg_template.σ2s,
                β=cfg_template.β, s=cfg_template.s,
                n_seed=cfg_template.n_seed, steps=cfg_template.steps,
                tune_every=cfg_template.tune_every, R_diag_init=cfg_template.R_diag_init,
                animate=false, log_every=cfg_template.log_every, seed=seed,
                obs_pattern=:full, obs_frac=1.0,
            )
            eval_fn = eval_fn_factory(d)

            setup_data = setup_experiment(cfg, eval_fn)
            chain      = nn_chain_quality(setup_data.Xo)

            @info "Running SS-GP (d=$d, seed=$seed)"
            po_state = setup_po(cfg, setup_data)
            out_ss = run_bo_po!(cfg, eval_fn;
                po_state=po_state, Xo=setup_data.Xo, Δ=setup_data.Δ,
                Ytrue=setup_data.Ytrue)

            @info "Running KM-GP (d=$d, seed=$seed)"
            setup_data2 = setup_experiment(cfg, eval_fn)
            mask_full   = _generate_obs_mask(cfg, cfg.N, MersenneTwister(seed + 1000))
            bl_state    = setup_baseline_po(cfg, setup_data2, mask_full)
            out_km = run_bo_baseline_po!(cfg, eval_fn;
                bl_state=bl_state, Ytrue=setup_data2.Ytrue)

            @info "Running Random (d=$d, seed=$seed)"
            setup_data3 = setup_experiment(cfg, eval_fn)
            out_rand = _run_random_bo(cfg, eval_fn;
                Xo=setup_data3.Xo, setup_data=setup_data3)

            push!(all_results, Dict(
                "d"     => d,
                "seed"  => seed,
                "chain" => Dict(
                    "mean_delta"   => chain.mean_delta,
                    "median_delta" => chain.median_delta,
                    "max_delta"    => chain.max_delta,
                    "min_delta"    => chain.min_delta,
                    "total_length" => chain.total_length,
                ),
                "ss" => Dict(
                    "best_value_history" => out_ss.result.best_value_history,
                    "step_times"         => out_ss.result.step_times,
                    "best_value"         => out_ss.result.best_value,
                    "n_iterations"       => out_ss.result.n_iterations,
                    "total_time"         => sum(out_ss.result.step_times),
                ),
                "km" => Dict(
                    "best_value_history" => out_km.result.best_value_history,
                    "step_times"         => out_km.result.step_times,
                    "best_value"         => out_km.result.best_value,
                    "n_iterations"       => out_km.result.n_iterations,
                    "total_time"         => sum(out_km.result.step_times),
                ),
                "random" => Dict(
                    "best_value_history" => out_rand.best_value_history,
                    "step_times"         => out_rand.step_times,
                    "best_value"         => out_rand.best_value,
                    "n_iterations"       => out_rand.n_iterations,
                    "total_time"         => sum(out_rand.step_times),
                ),
            ))
        end
    end

    json_path = joinpath(output_dir, "comparison.json")
    open(json_path, "w") do io
        JSON.print(io, all_results, 2)
    end
    @info "Saved dim_sweep comparison to $json_path"

    _plot_dim_sweep(all_results, ds, output_dir)

    all_results
end

# ─── Plotting ───────────────────────────────────────────────────────────────

"""
    _plot_dim_sweep(results, ds, output_dir)

Render the dim_sweep comparison figures: per-d convergence curves, final
regret vs d, total runtime vs d, and chain-quality vs d.
"""
function _plot_dim_sweep(results, ds, output_dir)
    methods = ["ss", "km", "random"]
    labels  = ["SS-GP", "KM-GP", "Random"]
    colors  = [:blue, :red, :gray]

    _nanmean(x)   = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
    _nanstd(x)    = (v = filter(!isnan, x); length(v) > 1 ? std(v) : 0.0)
    _nanmedian(x) = (v = filter(!isnan, x); isempty(v) ? NaN : median(v))

    function _pad(v, n)
        length(v) >= n && return v[1:n]
        vcat(v, fill(NaN, n - length(v)))
    end

    theme_kw = publication_theme_kwargs()

    # Per-d convergence curves
    for d in ds
        runs = filter(r -> r["d"] == d, results)
        isempty(runs) && continue
        max_steps = maximum(maximum(length(r[m]["best_value_history"]) for m in methods) for r in runs)
        save_plot(joinpath(output_dir, "convergence_d$(d)")) do
            p = plot(; xlabel="BO Step", ylabel="Best Scalarized Value",
                     title="d = $d", legend=:bottomright, theme_kw...)
            for (m, lab, col) in zip(methods, labels, colors)
                hist = hcat([_pad(r[m]["best_value_history"], max_steps) for r in runs]...)
                m_mean = [_nanmean(hist[i, :]) for i in 1:max_steps]
                m_std  = [_nanstd(hist[i, :])  for i in 1:max_steps]
                plot!(p, 1:max_steps, m_mean, ribbon=m_std, fillalpha=0.15, lw=2,
                      label=lab, color=col)
            end
            p
        end
    end

    # Final regret vs d
    save_plot(joinpath(output_dir, "final_regret_vs_d")) do
        p = plot(; xlabel="Input dimension d", ylabel="Best Scalarized Value (final)",
                 legend=:bottomleft, xscale=:log2, theme_kw...)
        for (m, lab, col) in zip(methods, labels, colors)
            means = Float64[]; stds = Float64[]
            for d in ds
                runs = filter(r -> r["d"] == d, results)
                vals = [r[m]["best_value"] for r in runs]
                push!(means, _nanmean(vals))
                push!(stds, _nanstd(vals))
            end
            plot!(p, ds, means, ribbon=stds, fillalpha=0.15, lw=2, marker=:circle,
                  label=lab, color=col)
        end
        p
    end

    # Total time per run vs d
    save_plot(joinpath(output_dir, "time_vs_d")) do
        p = plot(; xlabel="Input dimension d", ylabel="Median total wall-clock (s)",
                 legend=:topleft, xscale=:log2, yscale=:log10, theme_kw...)
        for (m, lab, col) in zip(methods, labels, colors)
            meds = Float64[]
            for d in ds
                runs = filter(r -> r["d"] == d, results)
                vals = [r[m]["total_time"] for r in runs]
                push!(meds, _nanmedian(vals))
            end
            plot!(p, ds, meds, lw=2, marker=:circle, label=lab, color=col)
        end
        p
    end

    # Chain quality vs d
    save_plot(joinpath(output_dir, "chain_quality_vs_d")) do
        p = plot(; xlabel="Input dimension d", ylabel="Chain Δ (consecutive distance)",
                 legend=:topleft, xscale=:log2, theme_kw...)
        mean_curve = Float64[]; mean_std = Float64[]
        max_curve  = Float64[]; max_std  = Float64[]
        for d in ds
            runs = filter(r -> r["d"] == d, results)
            mvals = [r["chain"]["mean_delta"] for r in runs]
            xvals = [r["chain"]["max_delta"]  for r in runs]
            push!(mean_curve, _nanmean(mvals)); push!(mean_std, _nanstd(mvals))
            push!(max_curve,  _nanmean(xvals)); push!(max_std,  _nanstd(xvals))
        end
        plot!(p, ds, mean_curve, ribbon=mean_std, fillalpha=0.15, lw=2,
              marker=:circle, label="mean Δ", color=:purple)
        plot!(p, ds, max_curve, ribbon=max_std, fillalpha=0.15, lw=2,
              marker=:diamond, label="max Δ", color=:orange, linestyle=:dash)
        p
    end
end
