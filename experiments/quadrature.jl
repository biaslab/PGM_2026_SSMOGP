# Bayesian Quadrature — sequential experimental design
#
# 1D synthetic benchmark (sinusoidal latent functions)
# d=1 input, D=6 outputs — NN chain is exact in 1D
#
# Compares 2 methods across multiple seeds:
#   SS-GP (full obs) vs KM-GP (full obs)
#
# Acquisition: max total predictive variance
# Metric: integral error ‖Z_hat - Z_true‖₂
#
# Run via: julia --project=. experiments/quadrature.jl
# Or via:  dvc repro quadrature

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using YAML, JSON, Statistics

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
p = params["quadrature"]

# Create 1D synthetic benchmark
eval_fn = make_synthetic_1d(D=p["D"], Q=p["Q"])

# Build template config
cfg = ExperimentConfig(
    N           = p["N"],
    d           = p["d"],
    Q           = p["Q"],
    D           = p["D"],
    ℓs          = Float64.(p["ls"]),
    σ2s         = Float64.(p["sigma2s"]),
    β           = 2.0,
    s           = fill(1.0 / p["D"], p["D"]),
    n_seed      = p["n_seed"],
    steps       = p["steps"],
    R_diag_init = Float64(p["R_diag_init"]),
    animate     = false,
    log_every   = p["log_every"],
    seed        = 0,
)

print_config(cfg)

output_dir = joinpath(@__DIR__, "..", "data", "quadrature")
seeds = p["seeds"]

results = run_bq_comparison(cfg, eval_fn; seeds=seeds, output_dir=output_dir)

# Save DVC metrics
metrics = Dict(
    "n_seeds"              => length(seeds),
    "n_steps"              => p["steps"],
    "ss_full_ie_final"     => mean([r["ss_full"]["ie_final"] for r in results]),
    "km_full_ie_final"     => mean([r["km_full"]["ie_final"] for r in results]),
    "ss_full_mnll_final"   => mean([r["ss_full"]["mnll_final"] for r in results]),
    "km_full_mnll_final"   => mean([r["km_full"]["mnll_final"] for r in results]),
    "ss_full_time_mean"    => mean([r["ss_full"]["total_time"] for r in results]),
    "km_full_time_mean"    => mean([r["km_full"]["total_time"] for r in results]),
)

open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.print(io, metrics, 2)
end
@info "Bayesian quadrature experiment complete" metrics
