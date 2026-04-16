"""
    setup_experiment(cfg::ExperimentConfig, eval_fn) -> NamedTuple

Prepare data for Bayesian optimization.

1. Generate `N` random points in `d` dimensions, standardize inputs
2. Compute nearest-neighbor chain ordering and inter-point distances `Δ`
3. Evaluate seed observations, standardize outputs
4. Build initial state-space GP blocks

Returns a named tuple with:
- `Xo`: ordered input matrix (N × d)
- `Δ`: normalized inter-point distances (length N)
- `blocks`: state-space GP matrices (A, Q, P, H)
- `W`: mixing matrix (D × Q)
- `Y`: observations vector with seed data (standardized)
- `μy`: mean used for standardization
- `σy`: std used for standardization
- `Ytrue`: ground-truth outputs for all N points (for visualization)
- `dist_norm`: median chain distance used for normalization
"""
function setup_experiment(cfg::ExperimentConfig, eval_fn)
    rng = MersenneTwister(cfg.seed)

    X = randn(rng, cfg.N, cfg.d)
    μx = vec(mean(X, dims=1))
    σx = vec(std(X, dims=1)) .+ eps()
    X = (X .- μx') ./ σx'

    order = cfg.d == 1 ? sortperm(X[:, 1]) : nn_chain_order(X)
    Xo = X[order, :]

    Δ = zeros(cfg.N)
    for i in 2:cfg.N
        Δ[i] = sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    dist_norm = median(Δ[2:end])
    Δ ./= dist_norm

    W = randn(rng, cfg.D, cfg.Q) .* 0.5
    blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs=cfg.ℓs, σ2s=cfg.σ2s, W)

    Y = Vector{Union{Missing,Vector{Float64}}}(undef, cfg.N)
    fill!(Y, missing)

    seed_idx = sort(rand(rng, 1:cfg.N, cfg.n_seed))
    Y_seed = [eval_fn(row(Xo, k)) for k in seed_idx]
    Y_seed_mat = hcat(Y_seed...)'

    μy = vec(mean(Y_seed_mat, dims=1))
    σy = vec(std(Y_seed_mat, dims=1)) .+ 1e-8

    for (i, k) in enumerate(seed_idx)
        Y[k] = (Y_seed[i] .- μy) ./ σy
    end

    Ytrue = [eval_fn(row(Xo, i)) for i in 1:cfg.N]

    (; Xo, Δ, blocks, W, Y, μy, σy, Ytrue, dist_norm)
end

"""
    _current_best(state::AbstractBOState, cfg::ExperimentConfig) -> (index, value, y)

Find the currently best observed point by scalarized value (original scale).
"""
function _current_best(state::AbstractBOState, cfg::ExperimentConfig)
    observed = findall(!ismissing, state.Y)
    best_idx = 0
    best_val = -Inf
    best_y = Float64[]
    for k in observed
        y_orig = state.Y[k] .* state.σy .+ state.μy
        val = dot(cfg.s, y_orig)
        if val > best_val
            best_val = val
            best_idx = k
            best_y = y_orig
        end
    end
    (; index=best_idx, value=best_val, y=best_y)
end

"""
    save_results(result::BOResult, cfg::ExperimentConfig, frames; output_dir="data")

Save experiment outputs to `output_dir/`:
- `metrics.json`: performance metrics for paper figures
- `bo_last_state.png`: snapshot of final BO step
- `bo_animation.gif`: animated GIF of all steps
"""
function save_results(result::BOResult, cfg::ExperimentConfig, frames; output_dir="data")
    mkpath(output_dir)

    # JSON performance metrics
    metrics = Dict(
        "method" => result.method,
        "best_index" => result.best_index,
        "best_value" => result.best_value,
        "best_y" => result.best_y,
        "n_iterations" => result.n_iterations,
        "n_observed" => length(result.observed_indices),
        "R_learned_diag" => diag(result.R_learned),
        "best_value_history" => result.best_value_history,
        "n_observed_history" => result.n_observed_history,
        "R_diag_history" => result.R_diag_history,
        "step_times" => result.step_times,
        "total_time" => sum(result.step_times),
        "config" => Dict(
            "N" => cfg.N, "d" => cfg.d, "Q" => cfg.Q, "D" => cfg.D,
            "beta" => cfg.β, "n_seed" => cfg.n_seed, "steps" => cfg.steps,
            "tune_every" => cfg.tune_every, "R_diag_init" => cfg.R_diag_init,
            "seed" => cfg.seed, "ls" => cfg.ℓs, "sigma2s" => cfg.σ2s, "s" => cfg.s
        )
    )
    json_path = joinpath(output_dir, "metrics.json")
    open(json_path, "w") do io
        JSON.json(io, metrics; pretty=2, allownan=true)
    end
    @info "Saved metrics to $json_path"

    # Last BO state (PNG + TikZ)
    if !isnothing(frames) && !isempty(frames)
        last_frame = last(frames)
        savefig(last_frame, joinpath(output_dir, "bo_last_state.png"))
        @info "Saved $(joinpath(output_dir, "bo_last_state")).png"
    end

    # Animated GIF
    if !isnothing(frames) && !isempty(frames)
        gif_path = joinpath(output_dir, "bo_animation.gif")
        anim = Animation()
        for plt in frames
            frame(anim, plt)
        end
        gif(anim, gif_path, fps=2)
        @info "Saved animation to $gif_path"
    end
end
