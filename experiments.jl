using Pkg; Pkg.activate(@__DIR__)

include("src/RxBayesOpt.jl")
using .RxBayesOpt

function eval_blackbox(x::AbstractVector{<:Real})
    y = zeros(10)
    y[1] = (x[1] + 0.5 * x[2] - 0.3 * x[3])^2 + 0.2 * sin(3 * x[1])
    y[2] = (x[4] - 0.4 * x[5] + 0.6 * x[6])^2 + 0.15 * cos(x[4])
    y[3] = x[7] * x[8] + 0.3 * x[9]^2 + 0.1 * sum(x[1:5])
    y[4] = sin(x[10] + x[11]) + 0.5 * x[12] * x[13]
    y[5] = x[14]^3 - 0.5 * x[15]^2 + 0.2 * x[16]
    y[6] = (x[17] + x[18])^2 + 0.3 * abs(x[19])
    y[7] = 0.5 * sum(x[1:10]) + 0.2 * sin(sum(x[11:15]))
    y[8] = exp(-0.1 * (x[20]^2 + x[1]^2)) + 0.3 * x[7]
    y[9] = 0.3 * sum(x[i]^2 for i in 1:10) + 0.1 * sum(x)
    y[10] = cos(0.5 * sum(x[11:20])) + 0.2 * sum(x[1:5] .* x[6:10])
    y
end

cfg = ExperimentConfig()
print_config(cfg)

(; Xo, Δ, state, Ytrue) = setup_experiment(cfg, eval_blackbox)
result = run_bo!(cfg, eval_blackbox; Xo, Δ, state, Ytrue)

print_summary(result)
