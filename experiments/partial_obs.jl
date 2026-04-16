# Partial observations — FFG modularity demonstration
#
# Environmental monitoring benchmark (Bliznyuk et al. 2008, Maddox et al. NeurIPS 2021)
# d=4 inputs, D=12 outputs (3 spatial × 4 temporal locations)
#
# Compares 4 methods across multiple seeds:
#   SS-GP (full obs), SS-GP (partial obs), KM-GP (full obs), KM-GP (partial obs)
#
# With D=12, KM-GP builds a (12×n_obs)² kernel matrix — O(M³) cost becomes
# significant as n_obs grows. SS-GP stays O(N) regardless.
#
# Run via: julia --project=. experiments/partial_obs.jl
# Or via:  dvc repro partial_obs

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using YAML, JSON, Statistics

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
p = params["partial_obs"]

# Create environmental monitoring benchmark
eval_fn = make_environmental(n_spatial=p["n_spatial"], n_temporal=p["n_temporal"])

# Build template config
cfg = ExperimentConfig(
    N           = p["N"],
    d           = p["d"],
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
    obs_pattern = :sensor_groups,
)

print_config(cfg)

output_dir = joinpath(@__DIR__, "..", "data", "partial_obs")
seeds = p["seeds"]

results = run_po_comparison(cfg, eval_fn; seeds=seeds, output_dir=output_dir)

# Helpers: last finite entry; nan-safe median/mean. Transient RxInfer numerical
# blowups leave NaNs in per-seed histories — we skip them rather than poisoning the summary.
_last_finite(v) = (f = filter(isfinite, v); isempty(f) ? NaN : f[end])
_nanmedian(x)   = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))
_nanmean(x)     = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))

# Save DVC metrics — median for quality metrics (robust to transient RxInfer instabilities),
# mean for best_value and timing.
metrics = Dict(
    "n_seeds"                => length(seeds),
    "n_steps"                => p["steps"],
    "ss_full_best_mean"      => _nanmean([r["ss_full"]["best_value"] for r in results]),
    "ss_po_best_mean"        => _nanmean([r["ss_po"]["best_value"] for r in results]),
    "km_full_best_mean"      => _nanmean([r["km_full"]["best_value"] for r in results]),
    "km_po_best_mean"        => _nanmean([r["km_po"]["best_value"] for r in results]),
    "random_best_mean"       => _nanmean([r["random"]["best_value"] for r in results]),
    "ss_full_time_mean"      => mean([r["ss_full"]["total_time"] for r in results]),
    "ss_po_time_mean"        => mean([r["ss_po"]["total_time"] for r in results]),
    "km_full_time_mean"      => mean([r["km_full"]["total_time"] for r in results]),
    "km_po_time_mean"        => mean([r["km_po"]["total_time"] for r in results]),
    "ss_full_rmse_final"     => _nanmedian([_last_finite(r["ss_full"]["rmse_history"]) for r in results]),
    "ss_po_rmse_final"       => _nanmedian([_last_finite(r["ss_po"]["rmse_history"]) for r in results]),
    "km_full_rmse_final"     => _nanmedian([_last_finite(r["km_full"]["rmse_history"]) for r in results]),
    "km_po_rmse_final"       => _nanmedian([_last_finite(r["km_po"]["rmse_history"]) for r in results]),
    "ss_full_mnll_final"     => _nanmedian([_last_finite(r["ss_full"]["mnll_history"]) for r in results]),
    "ss_po_mnll_final"       => _nanmedian([_last_finite(r["ss_po"]["mnll_history"]) for r in results]),
    "km_full_mnll_final"     => _nanmedian([_last_finite(r["km_full"]["mnll_history"]) for r in results]),
    "km_po_mnll_final"       => _nanmedian([_last_finite(r["km_po"]["mnll_history"]) for r in results]),
)

open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.json(io, metrics; pretty=2, allownan=true)
end
@info "Partial observation experiment complete" metrics
