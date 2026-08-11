# Input-dimension sweep on a synthetic sensor-network benchmark.
#
# Each input coordinate corresponds to one weather station. As the number of
# stations (d) grows, the NN-chain ordering used by SS-LMC becomes a weaker
# heuristic — this experiment quantifies the resulting held-out RMSE/MNLL gap
# against the exact kernel-matrix LMC (KM-LMC), which uses no chain ordering,
# plus two scalable approximations: SVGP-LMC (inducing points) and
# Vecchia/NNGP-LMC (nearest neighbours in the full M-dimensional input space).
#
# Run via: julia --project=. experiments/dim_sweep.jl
# Or via:  dvc repro dim_sweep

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using YAML, JSON, Statistics

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
p = params["dim_sweep"]

ds    = Int.(p["ds"])
seeds = Int.(p["seeds"])

cfg_template = ExperimentConfig(
    N           = p["N"],
    d           = first(ds),                   # placeholder; overwritten per-d in the sweep
    Q           = p["Q"],
    D           = p["D"],
    ℓs          = Float64.(p["ls"]),
    σ2s         = Float64.(p["sigma2s"]),
    β           = Float64(p["beta"]),
    s           = fill(1.0 / p["D"], p["D"]),
    n_seed      = p["n_seed"],
    steps       = p["steps"],
    R_diag_init = Float64(p["R_diag_init"]),
    animate     = false,
    log_every   = p["log_every"],
    seed        = 0,
    obs_pattern = :full,
)

print_config(cfg_template)

output_dir = joinpath(@__DIR__, "..", "data", "dim_sweep")

train_frac = Float64(get(p, "train_frac", 0.5))

eval_fn_factory = d -> make_sensor_network(; d=d, D=p["D"])
results = run_dim_sweep(cfg_template, eval_fn_factory;
                       ds=ds, seeds=seeds, train_frac=train_frac,
                       M_svgp=Int(get(p, "M_svgp", 64)),
                       m_vecchia=Int(get(p, "m_vecchia", 20)),
                       output_dir=output_dir)

# DVC metrics: per-d held-out accuracy aggregates per method
_nanmean(x) = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
metrics = Dict{String, Any}()
for d in ds
    runs = filter(r -> r["d"] == d, results)
    key  = "d=$d"
    metrics[key] = Dict(
        "ss_rmse_mean"     => _nanmean([r["ss"]["rmse"]   for r in runs]),
        "km_rmse_mean"     => _nanmean([r["km"]["rmse"]   for r in runs]),
        "svgp_rmse_mean"   => _nanmean([r["svgp"]["rmse"] for r in runs]),
        "vec_rmse_mean"    => _nanmean([r["vec"]["rmse"]  for r in runs]),
        "ss_mnll_mean"     => _nanmean([r["ss"]["mnll"]   for r in runs]),
        "km_mnll_mean"     => _nanmean([r["km"]["mnll"]   for r in runs]),
        "svgp_mnll_mean"   => _nanmean([r["svgp"]["mnll"] for r in runs]),
        "vec_mnll_mean"    => _nanmean([r["vec"]["mnll"]  for r in runs]),
        "ss_time_mean"     => _nanmean([r["ss"]["time"]   for r in runs]),
        "km_time_mean"     => _nanmean([r["km"]["time"]   for r in runs]),
        "svgp_time_mean"   => _nanmean([r["svgp"]["time"] for r in runs]),
        "vec_time_mean"    => _nanmean([r["vec"]["time"]  for r in runs]),
        "chain_mean_delta" => _nanmean([r["chain"]["mean_delta"] for r in runs]),
        "chain_max_delta"  => _nanmean([r["chain"]["max_delta"]  for r in runs]),
    )
end

open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.print(io, metrics, 2)
end
@info "Dim sweep experiment complete" metrics
