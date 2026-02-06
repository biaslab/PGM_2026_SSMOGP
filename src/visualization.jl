"""
    plot_bo_step(step, k, state, acq, Ytrue, cfg) -> Plot

Generate the 3-panel BO visualization for a single step.

Panels:
1. Scalarized prediction with uncertainty and next query point
2. UCB acquisition function
3. First 3 output dimensions (true vs predicted)

# Arguments
- `step`: current BO iteration number
- `k`: chain index of the next point to evaluate
- `state`: current `BOState`
- `acq`: named tuple from `ucb_acquisition` (ucb, μ_pred, σ_pred, μs, σs)
- `Ytrue`: ground-truth outputs for all N points
- `cfg`: `ExperimentConfig`
"""
function plot_bo_step(step::Int, k::Int, state::BOState, acq, Ytrue, cfg::ExperimentConfig)
    N = cfg.N
    s = cfg.s
    observed = findall(!ismissing, state.Y)

    ys_true = [dot(s, Ytrue[i]) for i in 1:N]

    p1 = plot(1:N, ys_true, lw=1, alpha=0.3, color=:black, label="true",
        xlabel="chain index", ylabel="sᵀy", title="Step $step: UCB Acquisition")
    plot!(p1, 1:N, acq.μs, ribbon=2 .* acq.σs, fillalpha=0.2, lw=2, label="μ ± 2σ")
    scatter!(p1, observed, [dot(s, Ytrue[i]) for i in observed], ms=5, color=:red, label="obs", alpha=0.7)
    vline!(p1, [k], linestyle=:dash, lw=2, color=:green, label="next")

    p2 = plot(1:N, acq.ucb, lw=2, xlabel="chain index", ylabel="UCB", title="Acquisition Function", legend=false)
    scatter!(p2, observed, acq.ucb[observed], ms=3, alpha=0.5)
    vline!(p2, [k], linestyle=:dash, lw=2, color=:red)

    colors = [:blue, :red, :green]
    p3 = plot(title="First 3 Outputs", xlabel="chain index", ylabel="value", legend=:topright)
    for j in 1:3
        y_true_j = [Ytrue[i][j] for i in 1:N]
        μj = [acq.μ_pred[i][j] for i in 1:N]
        σj = [acq.σ_pred[i][j] for i in 1:N]
        plot!(p3, 1:N, y_true_j, lw=1, alpha=0.3, linestyle=:dot, label="true y$j", color=colors[j])
        plot!(p3, 1:N, μj, ribbon=2 .* σj, fillalpha=0.15, lw=2, label="pred y$j", color=colors[j])
    end
    vline!(p3, [k], linestyle=:dash, lw=2, label="next", color=:black, alpha=0.5)

    plot(p1, p2, p3, layout=@layout([a; b; c]), size=(900, 1000))
end
