# Input-dimension sweep on the synthetic sensor-network benchmark.
#
# Measures posterior quality (held-out RMSE and MNLL) of the state-space LMC
# (SS-LMC) against the exact kernel-matrix LMC (KM-LMC) as the input dimension
# `d` grows. Each input coordinate is one weather station; as `d` increases the
# NN-chain ordering used by SS-LMC becomes a weaker heuristic, so this sweep
# probes how gracefully the chain approximation degrades.
#
# For each (d, seed) a single 50/50 train/test split is drawn, both methods run
# one inference pass, and held-out RMSE/MNLL are recorded alongside the raw
# NN-chain quality so any accuracy gap can be correlated with chain stretch.

"""
    _compute_rmse_test(μ_pred, Ytrue, test_idx, D) -> Float64

Root mean squared error restricted to held-out test points.

    RMSE = sqrt( 1/(|test|·D) Σ_{i∈test} Σ_d (μ_pred[i][d] - Ytrue[i][d])² )
"""
function _compute_rmse_test(μ_pred::Vector{Vector{Float64}}, Ytrue::Vector{Vector{Float64}},
                            test_idx::Vector{Int}, D::Int)
    mse = 0.0
    for i in test_idx
        for d in 1:D
            mse += (μ_pred[i][d] - Ytrue[i][d])^2
        end
    end
    sqrt(mse / (length(test_idx) * D))
end

"""
    _compute_mnll_obs(μ_pred, σ_pred, Ytrue, test_idx, σy, R_diag_init, D) -> Float64

Mean negative log predictive density of a held-out *observation* on the test set.

The predictive variance combines the GP latent-function variance `σ_pred[i][d]^2`
with the model's observation-noise variance, which in original output scale is
`R_diag_init * σy[d]^2` (the noise is defined on standardized outputs). Including
the noise floor makes the score well-posed for both methods: it prevents the NLL
from diverging where the state-space smoother's latent variance underflows to zero
at densely-chained test points, and treats SS-LMC and KM-LMC identically.

    MNLL = 1/(|test|·D) Σ_{i∈test} Σ_d [ ½ log(2π σ²ᵢd) + (yᵢd - μᵢd)² / (2 σ²ᵢd) ],
    σ²ᵢd = σ_pred[i][d]² + R_diag_init · σy[d]²
"""
function _compute_mnll_obs(μ_pred::Vector{Vector{Float64}}, σ_pred::Vector{Vector{Float64}},
                           Ytrue::Vector{Vector{Float64}}, test_idx::Vector{Int},
                           σy::Vector{Float64}, R_diag_init::Float64, D::Int)
    nll = 0.0
    for i in test_idx
        for d in 1:D
            σ2 = σ_pred[i][d]^2 + R_diag_init * σy[d]^2
            nll += 0.5 * log(2π * σ2) + (Ytrue[i][d] - μ_pred[i][d])^2 / (2σ2)
        end
    end
    nll / (length(test_idx) * D)
end

"""
    _train_test_split(setup_data, cfg, train_frac, rng) -> (; sd, train_idx, test_idx)

Draw a `train_frac` fraction of the `N` candidate points (without replacement) as
the observed/training set; the rest are held out for evaluation. Returns a copy of
`setup_data` (a NamedTuple) whose `Y`, `μy`, `σy` are overwritten so that training
points carry standardized observations and test points are `missing`. Output
standardization statistics are recomputed from the training outputs.
"""
function _train_test_split(setup_data, cfg::ExperimentConfig, train_frac::Float64, rng)
    N = cfg.N
    Ytrue = setup_data.Ytrue

    n_train = round(Int, train_frac * N)
    perm = randperm(rng, N)
    train_idx = sort(perm[1:n_train])
    test_idx  = sort(perm[n_train+1:end])

    # Standardize from training outputs only.
    Y_train_mat = hcat([Ytrue[k] for k in train_idx]...)'
    μy = vec(mean(Y_train_mat, dims=1))
    σy = vec(std(Y_train_mat, dims=1)) .+ 1e-8

    Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
    fill!(Y, missing)
    for k in train_idx
        Y[k] = (Ytrue[k] .- μy) ./ σy
    end

    sd = merge(setup_data, (; Y=Y, μy=μy, σy=σy))
    (; sd, train_idx, test_idx)
end

"""
    run_dim_sweep(cfg_template, eval_fn_factory; ds, seeds, train_frac, output_dir) -> Vector{Dict}

Sweep held-out posterior quality over input dimensions `ds`, comparing SS-LMC and
the exact kernel-matrix LMC (KM-LMC). `eval_fn_factory(d)` must return a function
whose input is a length-`d` vector. The other config fields (`D`, `Q`, `N`, etc.)
are held fixed.

For each (d, seed):
  1. Build a config with this `d` and `seed`.
  2. `setup_experiment` once, then draw a `train_frac` train/test split.
  3. SS-LMC: one `infer` pass; KM-LMC: one kernel-matrix prediction.
  4. Compute held-out RMSE and MNLL for both methods.
  5. Record `nn_chain_quality(Xo)` and per-method inference time.

Writes `comparison.json` and rendered figures to `output_dir`.
"""
function run_dim_sweep(cfg_template::ExperimentConfig, eval_fn_factory;
                      ds::Vector{Int}=[2, 4, 8, 16, 32], seeds=0:4,
                      train_frac::Float64=0.5,
                      M_svgp::Int=64,
                      output_dir::String="data/dim_sweep")
    mkpath(output_dir)
    all_results = Dict{String, Any}[]

    # Warm up JIT (RxInfer compilation + first-call autodiff dispatch) so the
    # first recorded (d, seed) cell isn't inflated by compile time. We discard
    # the result; cost is one extra inference pass at the smallest d.
    @info "Warming up (compilation)…"
    let d = first(ds)
        cfg = ExperimentConfig(;
            N=cfg_template.N, d=d, Q=cfg_template.Q, D=cfg_template.D,
            ℓs=cfg_template.ℓs, σ2s=cfg_template.σ2s,
            β=cfg_template.β, s=cfg_template.s,
            n_seed=cfg_template.n_seed, steps=cfg_template.steps,
            tune_every=cfg_template.tune_every, R_diag_init=cfg_template.R_diag_init,
            animate=false, log_every=cfg_template.log_every, seed=0,
            obs_pattern=:full, obs_frac=1.0,
        )
        sd_warm = setup_experiment(cfg, eval_fn_factory(d))
        split_warm = _train_test_split(sd_warm, cfg, train_frac, MersenneTwister(0))
        try
            po_warm = setup_po(cfg, split_warm.sd)
            res_warm = infer(
                model=additive_gp_po(
                    P=po_warm.blocks.P, A=po_warm.blocks.A,
                    Q=po_warm.blocks.Q, H=po_warm.blocks.H,
                    τ=po_warm.τ, e_vecs=po_warm.e_vecs, N=cfg.N, D=cfg.D),
                data=(Y=po_warm.Y_flat,),
                options=(limit_stack_depth=1000,))
            variance_acquisition(res_warm, po_warm, cfg, cfg.N)
            mask_warm = _generate_obs_mask(cfg, cfg.N, MersenneTwister(1000))
            baseline_po_variance_acquisition(setup_baseline_po(cfg, split_warm.sd, mask_warm), cfg)
            _lmc_svgp_predict(setup_svgp(cfg, split_warm.sd; M=M_svgp, Z_seed=9999))
        catch e
            @warn "Warmup pass failed (continuing anyway)" exception=e
        end
    end

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
            D = cfg.D
            N = cfg.N

            setup_data = setup_experiment(cfg, eval_fn)
            chain      = nn_chain_quality(setup_data.Xo)

            split = _train_test_split(setup_data, cfg, train_frac,
                                      MersenneTwister(seed + 7777))
            sd, test_idx = split.sd, split.test_idx

            # ── SS-LMC: single inference pass ──
            @info "Running SS-LMC (d=$d, seed=$seed)"
            po_state = setup_po(cfg, sd)
            ss_rmse = NaN; ss_mnll = NaN
            ss_time = @elapsed begin
                try
                    res = infer(
                        model=additive_gp_po(
                            P=po_state.blocks.P, A=po_state.blocks.A,
                            Q=po_state.blocks.Q, H=po_state.blocks.H,
                            τ=po_state.τ, e_vecs=po_state.e_vecs, N=N, D=D),
                        data=(Y=po_state.Y_flat,),
                        options=(limit_stack_depth=1000,))
                    acq = variance_acquisition(res, po_state, cfg, N)
                    ss_rmse = _compute_rmse_test(acq.μ_pred, sd.Ytrue, test_idx, D)
                    ss_mnll = _compute_mnll_obs(acq.μ_pred, acq.σ_pred, sd.Ytrue, test_idx,
                                                sd.σy, cfg.R_diag_init, D)
                catch e
                    @warn "SS-LMC inference failed (d=$d, seed=$seed)" exception=e
                end
            end

            # ── KM-LMC: single kernel-matrix prediction ──
            @info "Running KM-LMC (d=$d, seed=$seed)"
            mask_full = _generate_obs_mask(cfg, N, MersenneTwister(seed + 1000))
            bl_state  = setup_baseline_po(cfg, sd, mask_full)
            km_rmse = NaN; km_mnll = NaN
            km_time = @elapsed begin
                try
                    acq = baseline_po_variance_acquisition(bl_state, cfg)
                    km_rmse = _compute_rmse_test(acq.μ_pred, sd.Ytrue, test_idx, D)
                    km_mnll = _compute_mnll_obs(acq.μ_pred, acq.σ_pred, sd.Ytrue, test_idx,
                                                sd.σy, cfg.R_diag_init, D)
                catch e
                    @warn "KM-LMC prediction failed (d=$d, seed=$seed)" exception=e
                end
            end

            # ── SVGP-LMC: closed-form variational posterior at M inducing points ──
            # K-means inducing-point selection is part of the SVGP wall-clock budget.
            @info "Running SVGP (d=$d, seed=$seed)"
            svgp_rmse = NaN; svgp_mnll = NaN
            svgp_time = @elapsed begin
                try
                    svgp_state = setup_svgp(cfg, sd; M=M_svgp, Z_seed=seed + 9999)
                    pred = _lmc_svgp_predict(svgp_state)
                    # Rescale to original output scale (matches SS-LMC / KM-LMC paths).
                    μ_orig = [pred.μ_pred[i] .* sd.σy .+ sd.μy for i in 1:N]
                    σ_orig = [pred.σ_pred[i] .* sd.σy           for i in 1:N]
                    svgp_rmse = _compute_rmse_test(μ_orig, sd.Ytrue, test_idx, D)
                    svgp_mnll = _compute_mnll_obs(μ_orig, σ_orig, sd.Ytrue,
                                                  test_idx, sd.σy, cfg.R_diag_init, D)
                catch e
                    @warn "SVGP prediction failed (d=$d, seed=$seed)" exception=e
                end
            end

            @info "Result (d=$d, seed=$seed)" ss_rmse=round(ss_rmse; digits=4) km_rmse=round(km_rmse; digits=4) svgp_rmse=round(svgp_rmse; digits=4) ss_mnll=round(ss_mnll; digits=4) km_mnll=round(km_mnll; digits=4) svgp_mnll=round(svgp_mnll; digits=4)

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
                "ss"   => Dict("rmse" => ss_rmse,   "mnll" => ss_mnll,   "time" => ss_time),
                "km"   => Dict("rmse" => km_rmse,   "mnll" => km_mnll,   "time" => km_time),
                "svgp" => Dict("rmse" => svgp_rmse, "mnll" => svgp_mnll, "time" => svgp_time),
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

Render the dim_sweep figures: held-out RMSE vs d, held-out MNLL vs d,
fit+predict time vs d (log-y), and raw NN-chain quality vs d. Each plot
overlays SS-LMC, KM-LMC, and SVGP-LMC where applicable.
"""
function _plot_dim_sweep(results, ds, output_dir)
    methods = ["ss", "km", "svgp"]
    labels  = ["SS-LMC", "KM-LMC", "SVGP-LMC"]
    colors  = [:blue, :red, :green]

    _nanmean(x)   = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
    _nanstd(x)    = (v = filter(!isnan, x); length(v) > 1 ? std(v) : 0.0)

    theme_kw = publication_theme_kwargs()

    function _metric_vs_d(metric, ylabel, fname; legendpos=:topleft, yscale=:identity)
        save_plot(joinpath(output_dir, fname)) do
            p = plot(; xlabel="Input dimension d", ylabel=ylabel,
                     legend=legendpos, xscale=:log2, yscale=yscale, theme_kw...)
            for (m, lab, col) in zip(methods, labels, colors)
                means = Float64[]; stds = Float64[]
                for d in ds
                    runs = filter(r -> r["d"] == d, results)
                    vals = [r[m][metric] for r in runs]
                    push!(means, _nanmean(vals))
                    push!(stds, _nanstd(vals))
                end
                # On a log scale, a ribbon that crosses zero crashes the renderer;
                # clip it to a positive floor before drawing.
                ribbon_vals = yscale == :log10 ? min.(stds, means .* 0.99) : stds
                plot!(p, ds, means, ribbon=ribbon_vals, fillalpha=0.15, lw=2,
                      marker=:circle, label=lab, color=col)
            end
            p
        end
    end

    _metric_vs_d("rmse", "Held-out RMSE",        "rmse_vs_d")
    _metric_vs_d("mnll", "Held-out MNLL",        "mnll_vs_d")
    _metric_vs_d("time", "Fit+predict time (s)", "time_vs_d"; yscale=:log10)

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
