# ETT oil-temperature forecasting — partial observations on real data.
#
# Dataset: ETTh1 (Zhou et al. 2021), placed locally at data/ETTh1.csv.
# Setup:   d=1 input (time), D=4 outputs (HUFL, MUFL, LUFL, OT).
#          First train_frac of timestamps train, remainder test (fully masked).
#          Dropout ∈ {0%, 30%, 60%} on training half, per (timestamp, output).
# Metrics: MSE and NLL on OT only, restricted to test timestamps.
# Methods: SS-GP (partial obs) vs KM-GP (partial obs).
#
# Run via: julia --project=. experiments/ett_forecast.jl
# Or via:  dvc repro ett_forecast

using Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()

using YAML, JSON, Statistics

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
p = params["ett_forecast"]

# ── Load data ──────────────────────────────────────────────────────────────
cols = Tuple(String.(p["cols"]))
loaded = load_etth1(; N=p["N"], cols=cols)
X     = loaded.X
Y     = loaded.Y
ot_idx = loaded.ot_idx
@info "Loaded ETTh1" N=size(X,1) D=length(loaded.col_names) cols=loaded.col_names ot_idx

# ── Build template config ──────────────────────────────────────────────────
D = length(loaded.col_names)
cfg = ExperimentConfig(
    N           = p["N"],
    d           = 1,
    Q           = p["Q"],
    D           = D,
    ℓs          = Float64.(p["ls"]),
    σ2s         = Float64.(p["sigma2s"]),
    β           = 2.0,
    s           = fill(1.0 / D, D),
    n_seed      = 0,
    steps       = 0,
    R_diag_init = Float64(p["R_diag_init"]),
    animate     = false,
    log_every   = p["log_every"],
    seed        = 0,
)

print_config(cfg)

output_dir   = joinpath(@__DIR__, "..", "data", "ett_forecast")
seeds        = p["seeds"]
dropouts     = Float64.(p["dropouts"])
train_frac   = Float64(p["train_frac"])
n_test_steps = Int(p["n_test_steps"])

test_dropout_mode = Symbol(get(p, "test_dropout_mode", "same_as_train"))
horizons          = Int.(get(p, "horizons", [1]))
plot_horizon      = Int(get(p, "plot_horizon", maximum(horizons)))

results = run_ett_comparison(cfg, X, Y;
    seeds=seeds, dropouts=dropouts, horizons=horizons,
    train_frac=train_frac, n_test_steps=n_test_steps, d_target=ot_idx,
    test_dropout_mode=test_dropout_mode, plot_horizon=plot_horizon,
    output_dir=output_dir)

# ── DVC metrics: median (robust) and mean across seeds, plus diverged count ──
metrics = Dict{String, Any}(
    "n_seeds"    => length(seeds),
    "N"          => p["N"],
    "train_frac" => train_frac,
    "dropouts"   => dropouts,
)
for method in ("ss", "km"), metric in ("mse", "nll", "time")
    for dropout in dropouts, h in horizons
        # Exclude diverged SS-GP runs from aggregates so they don't dominate.
        all_vals = [r[method][metric] for r in results
                    if r["dropout"] == dropout && r["horizon"] == h]
        good_vals = [r[method][metric] for r in results
                     if r["dropout"] == dropout && r["horizon"] == h &&
                        !get(r[method], "diverged", false)]
        key_base = "$(method)_$(metric)_dp$(round(Int, 100*dropout))_h$(h)"
        if !isempty(good_vals)
            metrics["$(key_base)_median"] = median(good_vals)
            metrics["$(key_base)_mean"]   = mean(good_vals)
        else
            metrics["$(key_base)_median"] = NaN
            metrics["$(key_base)_mean"]   = NaN
        end
        metrics["$(key_base)_n_used"]     = length(good_vals)
        metrics["$(key_base)_n_diverged"] = length(all_vals) - length(good_vals)
    end
end

open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.print(io, metrics, 2)
end

@info "ETT forecasting experiment complete" output_dir
