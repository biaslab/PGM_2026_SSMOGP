"""
    publication_theme_kwargs(; single_column=true, nrows=1, ncols=1, conference=:ProbNum25)

Return a NamedTuple of Plots.jl keyword arguments for publication-quality figures
using the TuePlots.jl theme settings.

`font=false` because Plots.jl doesn't support it; PGFPlotsX handles fonts via LaTeX.
"""
function publication_theme_kwargs(; single_column=true, nrows=1, ncols=1, conference=:ProbNum25)
    TuePlots.get_plotsjl_theme_kwargs(
        TuePlots.SETTINGS[conference];
        font=false, fontsize=true, figsize=true,
        single_column=single_column, nrows=nrows, ncols=ncols,
        thickness_scaling=true,
    )
end

"""
    save_plot(plot_fn, base_path)

Save a plot as both PNG and TikZ (.tikz).
`plot_fn` is a zero-argument function that creates and returns a Plots.jl plot.
`base_path` should be without extension (e.g., "data/convergence").
"""
function save_plot(plot_fn::Function, base_path; tikz::Bool=true)
    Base.invokelatest(gr)
    plt = Base.invokelatest(plot_fn)
    Base.invokelatest(savefig, plt, base_path * ".png")
    @info "Saved $(base_path).png"

    tikz || return plt

    try
        Base.invokelatest(pgfplotsx)
        plt_tikz = Base.invokelatest(plot_fn)
        Base.invokelatest(savefig, plt_tikz, base_path * ".tikz")
        @info "Saved $(base_path).tikz"
    catch e
        @warn "Failed to save tikz for $base_path" exception=e
    finally
        Base.invokelatest(gr)
    end
    plt
end

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
- `state`: current `AbstractBOState`
- `acq`: named tuple from `ucb_acquisition` (ucb, μ_pred, σ_pred, μs, σs)
- `Ytrue`: ground-truth outputs for all N points
- `cfg`: `ExperimentConfig`
"""
function plot_bo_step(step::Int, k::Int, state::AbstractBOState, acq, Ytrue, cfg::ExperimentConfig)
    N = cfg.N
    s = cfg.s
    observed = findall(!ismissing, state.Y)
    theme_kw = publication_theme_kwargs(nrows=3)

    ys_true = [dot(s, Ytrue[i]) for i in 1:N]

    p1 = plot(1:N, ys_true, lw=1, alpha=0.3, color=:black, label="true",
        xlabel="chain index", ylabel="sᵀy")
    plot!(p1, 1:N, acq.μs, ribbon=2 .* acq.σs, fillalpha=0.2, lw=2, label="μ ± 2σ")
    scatter!(p1, observed, [dot(s, Ytrue[i]) for i in observed], ms=3, color=:red, label="obs", alpha=0.7)
    vline!(p1, [k], linestyle=:dash, lw=2, color=:green, label="next")

    p2 = plot(1:N, acq.ucb, lw=2, xlabel="chain index", ylabel="UCB", legend=false)
    scatter!(p2, observed, acq.ucb[observed], ms=2, alpha=0.5)
    vline!(p2, [k], linestyle=:dash, lw=2, color=:red)

    colors = [:blue, :red, :green]
    linestyles = [:solid, :dash, :dot]
    p3 = plot(xlabel="chain index", ylabel="value", legend=:topright)
    for j in 1:3
        y_true_j = [Ytrue[i][j] for i in 1:N]
        μj = [acq.μ_pred[i][j] for i in 1:N]
        σj = [acq.σ_pred[i][j] for i in 1:N]
        plot!(p3, 1:N, y_true_j, lw=1, alpha=0.3, linestyle=:dot, label="true y$j", color=colors[j])
        plot!(p3, 1:N, μj, ribbon=2 .* σj, fillalpha=0.15, lw=2, linestyle=linestyles[j], label="pred y$j", color=colors[j])
    end
    vline!(p3, [k], linestyle=:dash, lw=2, label="next", color=:black, alpha=0.5)

    plot(p1, p2, p3, layout=@layout([a; b; c]); theme_kw...)
end
