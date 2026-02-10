# Minimal SS-GP BO run for profiling
#
# Small problem: d=4, D=4, N=50, 5 BO steps, single seed.
# Only runs the RxInfer (state-space) path — no kernel-matrix baseline.
#
# Usage:
#   julia --project=. experiments/profile.jl
#
# Profiling (e.g. with Profile stdlib):
#   julia --project=. -e '
#     include("experiments/profile.jl")
#     using Profile
#     Profile.clear()
#     @profile run_profiling()
#     Profile.print(mincount=20, sortedby=:count)
#   '

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

# Small environmental benchmark: 2 spatial × 2 temporal = D=4
eval_fn = RxBayesOpt.make_environmental(n_spatial=2, n_temporal=2)

cfg = RxBayesOpt.ExperimentConfig(
    N           = 50,
    d           = 4,
    Q           = 2,
    D           = 4,
    ℓs          = [1.0, 1.5],
    σ2s         = [3.0, 2.0],
    β           = 2.0,
    s           = fill(0.25, 4),
    n_seed      = 3,
    steps       = 5,
    R_diag_init = 0.2,
    animate     = false,
    log_every   = 1,
    seed        = 42,
    obs_pattern = :sensor_groups,
)

function run_profiling()
    setup_data = RxBayesOpt.setup_experiment(cfg, eval_fn)
    po = RxBayesOpt.setup_po(cfg, setup_data)
    out = RxBayesOpt.run_bo_po!(cfg, eval_fn;
        po_state=po, Xo=setup_data.Xo, Δ=setup_data.Δ, Ytrue=setup_data.Ytrue)
    RxBayesOpt.print_summary(out.result)
    return out
end

# Warmup (JIT compilation)
@info "Warmup run..."
run_profiling()

# Timed run
@info "Profiling run..."
@time out = run_profiling()
