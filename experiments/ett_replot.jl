# Lightweight replot: regenerate the four aggregate plots from comparison.json
# and re-run JUST the plot-seed × plot-horizon setups for the predictions_ot
# plot. Avoids a full 90-min sweep when only the plot rendering has changed.
#
# Run via: julia --project=. experiments/ett_replot.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using YAML, JSON, Statistics, Random

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
p = params["ett_forecast"]

cols = Tuple(String.(p["cols"]))
loaded = load_etth1(; N=p["N"], cols=cols)
X      = loaded.X
Y      = loaded.Y
ot_idx = loaded.ot_idx
D      = length(loaded.col_names)

cfg = ExperimentConfig(
    N=p["N"], d=1, Q=p["Q"], D=D,
    ℓs=Float64.(p["ls"]), σ2s=Float64.(p["sigma2s"]),
    β=2.0, s=fill(1.0/D, D),
    n_seed=0, steps=0,
    R_diag_init=Float64(p["R_diag_init"]),
    animate=false, log_every=p["log_every"], seed=0,
)

output_dir = joinpath(@__DIR__, "..", "data", "ett_forecast")
dropouts     = Float64.(p["dropouts"])
horizons     = Int.(p["horizons"])
n_test_steps = Int(p["n_test_steps"])
train_frac   = Float64(p["train_frac"])
test_dropout_mode = Symbol(get(p, "test_dropout_mode", "same_as_train"))
plot_horizon = Int(get(p, "plot_horizon", maximum(horizons)))
seeds        = p["seeds"]
plot_seed    = first(seeds)

# ── Load existing comparison.json so aggregate plots use the cached metrics ──
results = JSON.parsefile(joinpath(output_dir, "comparison.json"))
@info "Loaded $(length(results)) records from comparison.json"

# ── Regenerate plot_payload by re-running just the plot_seed × dropouts × plot_horizon ──
h_max = maximum(horizons)
plot_payload = Dict{Float64, Any}()

for dropout in dropouts
    @info "Rebuilding plot_payload" seed=plot_seed dropout horizon=plot_horizon

    cfg_seed = ExperimentConfig(;
        N=cfg.N, d=cfg.d, Q=cfg.Q, D=cfg.D,
        ℓs=cfg.ℓs, σ2s=cfg.σ2s, β=cfg.β, s=cfg.s,
        n_seed=cfg.n_seed, steps=cfg.steps,
        tune_every=cfg.tune_every, R_diag_init=cfg.R_diag_init,
        animate=false, log_every=cfg.log_every, seed=plot_seed,
        obs_pattern=:full, obs_frac=cfg.obs_frac)

    rng_mask = MersenneTwister(plot_seed + 7000)
    base_mask, n_train = RxBayesOpt._make_rolling_mask(
        cfg_seed.N, cfg_seed.D, train_frac, dropout, n_test_steps + h_max - 1, rng_mask)
    base = RxBayesOpt._build_rolling_setup(cfg_seed, X, Y, base_mask)
    cursor_range = (n_train + 1):(n_train + n_test_steps)
    test_dropout = test_dropout_mode == :same_as_train ? dropout : 0.0

    out_ss = RxBayesOpt.run_one_step_ahead_po(cfg_seed, base, cursor_range, ot_idx;
        forecast_horizon=plot_horizon, test_dropout=test_dropout,
        test_mask_rng=MersenneTwister(plot_seed + 8000))
    out_km = RxBayesOpt.run_one_step_ahead_baseline_po(cfg_seed, base, cursor_range, ot_idx;
        forecast_horizon=plot_horizon, test_dropout=test_dropout,
        test_mask_rng=MersenneTwister(plot_seed + 8000))

    ctx_start = max(1, n_train - 80)
    ctx_end   = min(cfg_seed.N, n_train + n_test_steps + plot_horizon - 1)
    ctx_idx   = ctx_start:ctx_end

    plot_payload[dropout] = (;
        ctx_x    = base.Xo[ctx_idx, 1],
        ctx_y    = [base.Y_ordered[i][ot_idx] for i in ctx_idx],
        ctx_idx  = collect(ctx_idx),
        train_obs_x = [base.Xo[i, 1] for i in ctx_start:n_train
                       if base.base_mask[i, ot_idx]],
        train_obs_y = [base.Y_ordered[i][ot_idx] for i in ctx_start:n_train
                       if base.base_mask[i, ot_idx]],
        test_x   = out_km.x_t, test_y = out_km.y_t,
        ss_μ = out_ss.μ_t, ss_σ = out_ss.σ_t, ss_diverged = out_ss.diverged,
        km_μ = out_km.μ_t, km_σ = out_km.σ_t,
        ss_full_μ = out_ss.full_μ, ss_full_σ = out_ss.full_σ,
        km_full_μ = out_km.full_μ, km_full_σ = out_km.full_σ,
        n_train  = n_train, horizon = plot_horizon,
    )
end

# ── Render all plots ──
RxBayesOpt._plot_ett_forecast(results, dropouts, horizons, output_dir)
RxBayesOpt._plot_ett_mse_vs_horizon(results, dropouts, horizons, output_dir)
RxBayesOpt._plot_ett_predictions(plot_payload, dropouts, output_dir; horizon=plot_horizon)
RxBayesOpt._plot_ett_timing(results, dropouts, horizons, output_dir)

@info "Replot complete" output_dir
