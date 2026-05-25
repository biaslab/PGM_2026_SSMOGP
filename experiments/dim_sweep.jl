# Input-dimension sweep on a synthetic sensor-network benchmark.
#
# Each input coordinate corresponds to one weather station. As the number of
# stations (d) grows, the NN-chain ordering used by SS-GP becomes a weaker
# heuristic — this experiment quantifies the resulting BO regret gap against
# a kernel-matrix GP (no chain ordering) and a random-acquisition baseline.
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

eval_fn_factory = d -> make_sensor_network(; d=d, D=p["D"])
results = run_dim_sweep(cfg_template, eval_fn_factory;
                       ds=ds, seeds=seeds, output_dir=output_dir)

# DVC metrics: per-d aggregates per method
metrics = Dict{String, Any}()
for d in ds
    runs = filter(r -> r["d"] == d, results)
    key  = "d=$d"
    metrics[key] = Dict(
        "ss_best_mean"     => mean([r["ss"]["best_value"]      for r in runs]),
        "km_best_mean"     => mean([r["km"]["best_value"]      for r in runs]),
        "random_best_mean" => mean([r["random"]["best_value"]  for r in runs]),
        "ss_time_mean"     => mean([r["ss"]["total_time"]      for r in runs]),
        "km_time_mean"     => mean([r["km"]["total_time"]      for r in runs]),
        "random_time_mean" => mean([r["random"]["total_time"]  for r in runs]),
        "chain_mean_delta" => mean([r["chain"]["mean_delta"]   for r in runs]),
        "chain_max_delta"  => mean([r["chain"]["max_delta"]    for r in runs]),
    )
end

open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.print(io, metrics, 2)
end
@info "Dim sweep experiment complete" metrics
