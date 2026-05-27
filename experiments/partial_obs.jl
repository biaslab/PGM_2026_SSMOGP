# Partial observations — message passing vs covariance restructuring (ETTh1 forecasting)
#
# Train a multi-output state-space LMC GP on the first half of the ETTh1 series with
# random per-feature dropout, then forecast the held-out second half. Compares:
#   SS-LMC (reactive message passing)  vs  KM-LMC (covariance restructuring)
#
# KM-LMC builds the full structured (D·N_train)² LMC kernel, slices the observed
# (time, feature) sub-block, and factorizes — an O((D·N_train)³) fit. SS-LMC handles
# the missing entries natively in O(N) message passing. Two sweeps: window size N and
# dropout level p; metrics: forecast MNLL, RMSE, fit+forecast wall-clock time.
#
# Run via: julia --project=. experiments/partial_obs.jl
# Or via:  dvc repro partial_obs

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using YAML, JSON, Statistics

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
p = params["partial_obs"]

data_path = joinpath(@__DIR__, "..", p["data_path"])
max_rows = maximum(Int.(p["Ns"]))
data = load_ett(data_path; n_rows=max_rows)
@info "Loaded ETT data" size=size(data) path=p["data_path"]

output_dir = joinpath(@__DIR__, "..", "data", "partial_obs")

res = run_ett_sweeps(data;
    Ns          = Int.(p["Ns"]),
    ps          = Float64.(p["ps"]),
    N_fixed     = Int(p["N_fixed"]),
    p_fixed     = Float64(p["p_fixed"]),
    seeds       = Int.(p["seeds"]),
    D           = Int(p["D"]),
    Q           = Int(p["Q"]),
    ℓs          = Float64.(p["ls"]),
    σ2s         = Float64.(p["sigma2s"]),
    R_diag_init = Float64(p["R_diag_init"]),
    output_dir  = output_dir,
)

# DVC metrics: mean over the largest-window cell of the N sweep
_mean(rows, m, metric) = mean(Float64(r[m][metric]) for r in rows)
big = [r for r in res.sweep_N if r["N"] == maximum(Int.(p["Ns"]))]
metrics = Dict(
    "n_seeds"          => length(p["seeds"]),
    "N_max"            => maximum(Int.(p["Ns"])),
    "ss_mnll_at_Nmax"  => _mean(big, "ss", "mnll"),
    "km_mnll_at_Nmax"  => _mean(big, "km", "mnll"),
    "ss_rmse_at_Nmax"  => _mean(big, "ss", "rmse"),
    "km_rmse_at_Nmax"  => _mean(big, "km", "rmse"),
    "ss_time_at_Nmax"  => _mean(big, "ss", "time"),
    "km_time_at_Nmax"  => _mean(big, "km", "time"),
)

open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.print(io, metrics, 2)
end
@info "ETT partial-observation forecasting experiment complete" metrics
