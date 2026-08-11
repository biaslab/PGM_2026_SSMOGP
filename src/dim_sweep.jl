# Input-dimension sweep on the synthetic sensor-network benchmark.
#
# Measures posterior quality (held-out RMSE and MNLL) of the state-space LMC
# (SS-LMC) against the exact kernel-matrix LMC (KM-LMC) as the input dimension
# `d` grows. Each input coordinate is one weather station; as `d` increases the
# NN-chain ordering used by SS-LMC becomes a weaker heuristic, so this sweep
# probes how gracefully the chain approximation degrades.
#
# Four methods share the same LMC prior (W, ℓs, σ2s, R):
#   ss   — SS-LMC, message passing along the 1-D nearest-neighbour chain
#   km   — KM-LMC, exact dense kernel matrix (the accuracy ceiling)
#   svgp — SVGP-LMC, M inducing points
#   vec  — Vecchia/NNGP-LMC, conditioning on m nearest neighbours in the full
#          M-dimensional input space (see `src/vecchia.jl`)
#
# `vec` is the sharpest comparison for this sweep: like SS-LMC it is built on a
# nearest-neighbour ordering, but it keeps the M-dimensional geometry instead of
# compressing it into one chain, so the ss-vs-vec gap isolates the cost of the
# chain compression specifically (rather than of sparsity in general).
#
# For each (d, seed) a single 50/50 train/test split is drawn, every method runs
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

Sweep held-out posterior quality over input dimensions `ds`, comparing SS-LMC,
the exact kernel-matrix LMC (KM-LMC), SVGP-LMC (`M_svgp` inducing points per
latent) and Vecchia/NNGP-LMC (`m_vecchia` nearest neighbouring locations).
`eval_fn_factory(d)` must return a function whose input is a length-`d` vector.
The other config fields (`D`, `Q`, `N`, etc.) are held fixed.

For each (d, seed):
  1. Build a config with this `d` and `seed`.
  2. `setup_experiment` once, then draw a `train_frac` train/test split.
  3. Run one inference pass per method against that identical split.
  4. Compute held-out RMSE and MNLL for every method.
  5. Record `nn_chain_quality(Xo)` and per-method inference time.

Writes `comparison.json` and rendered figures to `output_dir`.
"""
function run_dim_sweep(cfg_template::ExperimentConfig, eval_fn_factory;
                      ds::Vector{Int}=[2, 4, 8, 16, 32], seeds=0:4,
                      train_frac::Float64=0.5,
                      M_svgp::Int=64,
                      m_vecchia::Int=20,
                      output_dir::String="data/dim_sweep")
    mkpath(output_dir)
    all_results = Dict{String, Any}[]

    # Warm up JIT so the first recorded (d, seed) cell isn't inflated by compile
    # time. We discard the result; cost is one extra pass at the smallest d.
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
            ss_lmc_filter_smooth(po_warm.blocks.P, po_warm.blocks.A, po_warm.blocks.Q,
                                 po_warm.blocks.H, po_warm.τ, po_warm.Y_flat,
                                 cfg.N, cfg.D)
            mask_warm = _generate_obs_mask(cfg, cfg.N, MersenneTwister(1000))
            baseline_po_variance_acquisition(setup_baseline_po(cfg, split_warm.sd, mask_warm), cfg)
            _lmc_svgp_predict(setup_svgp(cfg, split_warm.sd; M=M_svgp, Z_seed=9999))
            _lmc_vecchia_predict(setup_vecchia(cfg, split_warm.sd; m=m_vecchia))
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

            # ── SS-LMC: one Kalman filter + RTS smoother pass ──
            # Uses the hand-coded smoother of `ss_lmc_raw.jl`, as `src/ett.jl` does,
            # rather than RxInfer's generic message passing. Same model and same LMC
            # prior; the difference is purely numerical. At low input dimension the
            # chain is tight, so A_i -> I and Q_i hits the PSD clamp in
            # `matern32_blocks_from_Δ`; the generic Gaussian updates lose precision
            # there (d=2, seed 4 returned RMSE 8.5e7, against 0.062 from this path,
            # with every other cell agreeing to 5 significant figures), whereas the
            # Joseph-form updates here stay symmetric and PSD. Timing the smoother
            # alone also makes this panel comparable to the ETT study, which reports
            # the same raw-Kalman cost.
            @info "Running SS-LMC (d=$d, seed=$seed)"
            po_state = setup_po(cfg, sd)
            ss_rmse = NaN; ss_mnll = NaN; ss_time = NaN
            try
                local pred
                ss_time = @elapsed begin
                    pred = ss_lmc_filter_smooth(po_state.blocks.P, po_state.blocks.A,
                                                po_state.blocks.Q, po_state.blocks.H,
                                                po_state.τ, po_state.Y_flat, N, D)
                end
                μ_ss = [pred.μ_pred[i] .* sd.σy .+ sd.μy for i in 1:N]
                σ_ss = [pred.σ_pred[i] .* sd.σy           for i in 1:N]
                ss_rmse = _compute_rmse_test(μ_ss, sd.Ytrue, test_idx, D)
                ss_mnll = _compute_mnll_obs(μ_ss, σ_ss, sd.Ytrue, test_idx,
                                            sd.σy, cfg.R_diag_init, D)
            catch e
                @warn "SS-LMC smoothing failed (d=$d, seed=$seed)" exception=e
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

            # ── Vecchia/NNGP-LMC: local kriging on m nearest observed neighbours ──
            # Neighbour search is part of the Vecchia wall-clock budget, matching
            # how K-means inducing selection is charged to SVGP.
            @info "Running Vecchia (d=$d, seed=$seed)"
            vec_rmse = NaN; vec_mnll = NaN
            vec_time = @elapsed begin
                try
                    vec_state = setup_vecchia(cfg, sd; m=m_vecchia)
                    pred = _lmc_vecchia_predict(vec_state)
                    # Rescale to original output scale (matches the other paths).
                    μ_orig = [pred.μ_pred[i] .* sd.σy .+ sd.μy for i in 1:N]
                    σ_orig = [pred.σ_pred[i] .* sd.σy           for i in 1:N]
                    vec_rmse = _compute_rmse_test(μ_orig, sd.Ytrue, test_idx, D)
                    vec_mnll = _compute_mnll_obs(μ_orig, σ_orig, sd.Ytrue,
                                                 test_idx, sd.σy, cfg.R_diag_init, D)
                catch e
                    @warn "Vecchia prediction failed (d=$d, seed=$seed)" exception=e
                end
            end

            @info "Result (d=$d, seed=$seed)" ss_rmse=round(ss_rmse; digits=4) km_rmse=round(km_rmse; digits=4) svgp_rmse=round(svgp_rmse; digits=4) vec_rmse=round(vec_rmse; digits=4) ss_mnll=round(ss_mnll; digits=4) km_mnll=round(km_mnll; digits=4) svgp_mnll=round(svgp_mnll; digits=4) vec_mnll=round(vec_mnll; digits=4)

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
                "vec"  => Dict("rmse" => vec_rmse,  "mnll" => vec_mnll,  "time" => vec_time),
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
fit+predict time vs d (log-y), raw NN-chain quality vs d, and the paired
SS-LMC-minus-KM-LMC gap in RMSE and MNLL vs d. The first three overlay
SS-LMC, KM-LMC, SVGP-LMC, and NNGP-LMC where applicable.
"""
function _plot_dim_sweep(results, ds, output_dir)
    all_methods = ["ss", "km", "svgp", "vec"]
    all_labels  = ["SS-LMC", "KM-LMC", "SVGP-LMC", "NNGP-LMC"]
    all_colors  = [:blue, :red, :green, :purple]
    # Keep only methods actually present, so an archived `comparison.json` from
    # before a baseline existed can be re-rendered without erroring.
    keep    = [i for i in eachindex(all_methods)
               if any(haskey(r, all_methods[i]) for r in results)]
    methods = all_methods[keep]; labels = all_labels[keep]; colors = all_colors[keep]

    _nanmean(x)   = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
    _nanstd(x)    = (v = filter(!isnan, x); length(v) > 1 ? std(v) : 0.0)

    # Match the ETT sweep figures' export style (src/ett.jl `_plot_ett_sweep`):
    # both figure sets are `\resizebox`d to the same width in the paper, so they
    # share one canvas and font scale via `SWEEP_PLOT_KW` (src/visualization.jl).
    plot_kw = SWEEP_PLOT_KW

    function _metric_vs_d(metric, ylabel, fname; legendpos=:topleft, yscale=:identity,
                          legend_cols=1)
        save_plot(joinpath(output_dir, fname)) do
            p = plot(; xlabel="Input dimension M", ylabel=ylabel,
                     legend=legendpos, legend_columns=legend_cols,
                     xscale=:log2, yscale=yscale, plot_kw...)
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

    # The paper composes these three panels into one float, so the legend goes on
    # the left-most panel only; repeating it three times just costs plot area.
    _metric_vs_d("rmse", "Held-out RMSE",        "rmse_vs_d")
    _metric_vs_d("mnll", "Held-out MNLL",        "mnll_vs_d"; legendpos=false)
    _metric_vs_d("time", "Fit+predict time (s)", "time_vs_d";
                 yscale=:log10, legendpos=false)

    # Approximation gap against the exact posterior, as SS-LMC minus KM-LMC.
    # The difference is taken *within* a seed before averaging: the difficulty of
    # a particular draw is common to both methods, so pairing cancels it and
    # leaves a far tighter band than differencing the per-M means would.
    function _gap_vs_d(metric, ylabel, fname)
        save_plot(joinpath(output_dir, fname)) do
            means = Float64[]; stds = Float64[]
            for d in ds
                runs = filter(r -> r["d"] == d, results)
                gaps = [r["ss"][metric] - r["km"][metric] for r in runs]
                push!(means, _nanmean(gaps)); push!(stds, _nanstd(gaps))
            end
            p = plot(; xlabel="Input dimension M", ylabel=ylabel, legend=false,
                     xscale=:log2, plot_kw...)
            hline!(p, [0.0], lw=1, ls=:dash, color=:gray, label="")
            plot!(p, ds, means, ribbon=stds, fillalpha=0.15, lw=2, marker=:circle,
                  label="", color=:blue)
            p
        end
    end
    # Short labels: at SWEEP_PLOT_KW's font size a longer ylabel overruns the
    # left margin and is clipped. The sign convention is given in the caption.
    _gap_vs_d("rmse", "RMSE gap", "rmse_gap_vs_d")
    _gap_vs_d("mnll", "MNLL gap", "mnll_gap_vs_d")

    # Chain quality vs d
    save_plot(joinpath(output_dir, "chain_quality_vs_d")) do
        p = plot(; xlabel="Input dimension M", ylabel="Chain Δ",
                 legend=:topleft, xscale=:log2, plot_kw...)
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
