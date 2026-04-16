# Chain starting-point ablation (SS-GP full only)
#
# Varies the NN chain starting index on the environmental benchmark and records the
# final-step RMSE/MNLL/best-value for SS-GP (full observations). Only SS-GP uses the
# chain ordering — KM-GP and random baselines are chain-independent and therefore
# excluded to keep wall-clock tractable.
#
# Addresses reviewer R2's request to quantify the sensitivity of the greedy
# nearest-neighbor chain to its starting point.
#
# Run via: julia --project=. experiments/chain_ablation.jl
# Or via:  dvc repro chain_ablation

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using YAML, JSON, Statistics

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
p = params["chain_ablation"]

eval_fn = make_environmental(n_spatial=p["n_spatial"], n_temporal=p["n_temporal"])

output_root = joinpath(@__DIR__, "..", "data", "chain_ablation")
mkpath(output_root)

seeds = p["seeds"]
start_indices = p["start_indices"]

# Filter transient RxInfer numerical artifacts the same way the plotting code does
# (src/partial_obs.jl:645, src/sequential_design.jl:501). Normal RMSE is ~0.3 and
# normal |MNLL| is ~0.5; finite-but-extreme values (e.g., 1e6 spikes in the Kalman
# smoother) are artifacts, not signal.
const QUALITY_THR = 10.0

_is_clean(v, thr) = isfinite(v) && abs(v) <= thr
_clean(x, thr)    = filter(v -> _is_clean(v, thr), x)

# Last per-step value in a history that is both finite and within magnitude threshold.
# Matches the semantics of _last_finite in experiments/partial_obs.jl:56 but with
# the magnitude filter added.
_last_clean(v, thr) = (f = _clean(v, thr); isempty(f) ? NaN : f[end])

_nanmedian(x) = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))
_nanmean(x)   = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
_nanstd(x)    = (v = filter(isfinite, x); length(v) < 2 ? NaN : std(v))

function make_cfg(start_idx, seed)
    ExperimentConfig(
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
        seed        = seed,
        obs_pattern = :full,
        start_idx   = start_idx,
    )
end

# Per-start summary statistics (one scalar per start_idx, aggregated over seeds)
rmse_per_start = Float64[]
mnll_per_start = Float64[]
best_per_start = Float64[]

# Full per-(start, seed) records for the comparison JSON
per_run_records = Dict{String, Any}[]

for start_idx in start_indices
    @info "=== Chain ablation: start_idx=$start_idx ==="
    final_rmse_over_seeds = Float64[]
    final_mnll_over_seeds = Float64[]
    final_best_over_seeds = Float64[]

    for seed in seeds
        cfg = make_cfg(start_idx, seed)
        setup_data = setup_experiment(cfg, eval_fn)
        po = setup_po(cfg, setup_data)
        out = run_bo_po!(cfg, eval_fn;
            po_state=po, Xo=setup_data.Xo, Δ=setup_data.Δ, Ytrue=setup_data.Ytrue)
        r = out.result

        push!(per_run_records, Dict(
            "start_idx"          => start_idx,
            "seed"               => seed,
            "best_value"         => r.best_value,
            "best_value_history" => r.best_value_history,
            "rmse_history"       => r.rmse_history,
            "mnll_history"       => r.mnll_history,
            "step_times"         => r.step_times,
            "total_time"         => sum(r.step_times),
        ))

        push!(final_rmse_over_seeds, _last_clean(r.rmse_history, QUALITY_THR))
        push!(final_mnll_over_seeds, _last_clean(r.mnll_history, QUALITY_THR))
        push!(final_best_over_seeds, r.best_value)
    end

    push!(rmse_per_start, _nanmedian(final_rmse_over_seeds))
    push!(mnll_per_start, _nanmedian(final_mnll_over_seeds))
    push!(best_per_start, _nanmean(final_best_over_seeds))
end

# Persist the full per-run records — useful for plots or deeper analysis if ever needed.
open(joinpath(output_root, "comparison.json"), "w") do io
    JSON.json(io, per_run_records; pretty=2, allownan=true)
end

metrics = Dict(
    "start_indices"                    => start_indices,
    "n_seeds"                          => length(seeds),
    "n_steps"                          => p["steps"],
    "ss_full_rmse_per_start"           => rmse_per_start,
    "ss_full_mnll_per_start"           => mnll_per_start,
    "ss_full_best_per_start"           => best_per_start,
    "ss_full_rmse_std_across_starts"   => _nanstd(rmse_per_start),
    "ss_full_mnll_std_across_starts"   => _nanstd(mnll_per_start),
    "ss_full_best_std_across_starts"   => _nanstd(best_per_start),
    "ss_full_rmse_median_across_starts"=> _nanmedian(rmse_per_start),
    "ss_full_mnll_median_across_starts"=> _nanmedian(mnll_per_start),
    "ss_full_best_mean_across_starts"  => _nanmean(best_per_start),
)

open(joinpath(output_root, "metrics.json"), "w") do io
    JSON.json(io, metrics; pretty=2, allownan=true)
end
@info "Chain ablation complete" metrics
