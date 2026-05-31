# ETT multi-dim-input regression under random per-feature dropout.
#
# Splits the 7 ETTh columns into a 3D input (useful loads HUFL, MUFL, LUFL)
# and a 4D output (HULL, MULL, LULL, OT). The first N rows are training
# candidates (with i.i.d. per-cell dropout on outputs); the next N rows are a
# fully held-out test set. Inputs are NN-chain ordered jointly so SS-LMC can
# exploit O(N) message passing on 3D inputs. Compares:
#   SS-LMC (reactive message passing)
#   KM-LMC (covariance restructuring — full structured (D·N)³ kernel)
#   SVGP-LMC (M inducing pts; O(N·M²) per fit)
# Two p-sweeps (at C=N_fixed and at C=N_fixed_big) plus one N-sweep.
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
# Each cell consumes 2·N rows (train + held-out test). Load enough for the
# largest cell in any sweep.
max_N    = maximum(Int.(p["Ns"]))
max_N    = max(max_N, Int(p["N_fixed"]), Int(p["N_fixed_big"]))
max_rows = 2 * max_N
data = load_ett(data_path; n_rows=max_rows)
@info "Loaded ETT data" size=size(data) path=p["data_path"]

output_dir = joinpath(@__DIR__, "..", "data", "partial_obs")

res = run_ett_sweeps(data;
    Ns          = Int.(p["Ns"]),
    ps          = Float64.(p["ps"]),
    N_fixed     = Int(p["N_fixed"]),
    N_fixed_big = Int(p["N_fixed_big"]),
    p_fixed     = Float64(p["p_fixed"]),
    seeds       = Int.(p["seeds"]),
    D           = Int(p["D"]),
    Q           = Int(p["Q"]),
    ℓs          = Float64.(p["ls"]),
    σ2s         = Float64.(p["sigma2s"]),
    R_diag_init = Float64(p["R_diag_init"]),
    input_cols  = Int.(p["input_cols"]),
    output_cols = Int.(p["output_cols"]),
    M           = Int(get(p, "M", 64)),
    output_dir  = output_dir,
)

# DVC metrics: mean over the largest-window cell of the N sweep.
# Filter NaN/Inf so the JSON write doesn't choke on degenerate cells.
function _mean(rows, m, metric)
    vals = Float64[Float64(r[m][metric]) for r in rows]
    finite = filter(isfinite, vals)
    isempty(finite) ? NaN : mean(finite)
end
big = [r for r in res.sweep_N if r["N"] == maximum(Int.(p["Ns"]))]
metrics = Dict(
    "n_seeds"             => length(p["seeds"]),
    "N_max"               => maximum(Int.(p["Ns"])),
    "N_fixed_big"         => Int(p["N_fixed_big"]),
    "M"                   => Int(get(p, "M", 64)),
    "ss_mnll_at_Nmax"     => _mean(big, "ss",   "mnll"),
    "km_mnll_at_Nmax"     => _mean(big, "km",   "mnll"),
    "svgp_mnll_at_Nmax"   => _mean(big, "svgp", "mnll"),
    "ss_rmse_at_Nmax"     => _mean(big, "ss",   "rmse"),
    "km_rmse_at_Nmax"     => _mean(big, "km",   "rmse"),
    "svgp_rmse_at_Nmax"   => _mean(big, "svgp", "rmse"),
    "ss_time_at_Nmax"     => _mean(big, "ss",   "time"),
    "km_time_at_Nmax"     => _mean(big, "km",   "time"),
    "svgp_time_at_Nmax"   => _mean(big, "svgp", "time"),
)

_safe(x) = (x isa AbstractFloat && !isfinite(x)) ? nothing : x
open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.print(io, Dict(k => _safe(v) for (k, v) in metrics), 2)
end
@info "ETT multi-dim partial-observation experiment complete" metrics
