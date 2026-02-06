"""
    setup_experiment(cfg::ExperimentConfig, eval_fn) -> (; Xo, Δ, state, Ytrue)

Prepare data for Bayesian optimization.

1. Generate `N` random points in `d` dimensions, standardize inputs
2. Compute nearest-neighbor chain ordering and inter-point distances `Δ`
3. Evaluate seed observations, standardize outputs
4. Build initial state-space GP blocks

Returns a named tuple with:
- `Xo`: ordered input matrix (N × d)
- `Δ`: normalized inter-point distances (length N)
- `state`: initialized `BOState`
- `Ytrue`: ground-truth outputs for all N points (for visualization)
"""
function setup_experiment(cfg::ExperimentConfig, eval_fn)
    rng = MersenneTwister(cfg.seed)

    X = randn(rng, cfg.N, cfg.d)
    μx = vec(mean(X, dims=1))
    σx = vec(std(X, dims=1)) .+ eps()
    X = (X .- μx') ./ σx'

    order = nn_chain_order(X)
    Xo = X[order, :]

    Δ = zeros(cfg.N)
    for i in 2:cfg.N
        Δ[i] = sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    Δ ./= median(Δ[2:end])

    W = randn(rng, cfg.D, cfg.Q) .* 0.5
    blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs=cfg.ℓs, σ2s=cfg.σ2s, W)

    # InverseWishart prior on observation noise covariance R
    # ν = D + 2 gives a proper prior with finite mean = Ψ / (ν - D - 1) = Ψ
    ν0 = cfg.D + 2
    Ψ0 = cfg.R_diag_init * I(cfg.D) * (ν0 - cfg.D - 1)
    R_prior = InverseWishart(ν0, Matrix(Hermitian(Ψ0)))

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

    state = BOState(blocks, W, R_prior, Y, μy, σy)
    (; Xo, Δ, state, Ytrue)
end

"""
    _current_best(state::BOState, cfg::ExperimentConfig) -> (index, value, y)

Find the currently best observed point by scalarized value (original scale).
"""
function _current_best(state::BOState, cfg::ExperimentConfig)
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
    run_bo!(cfg::ExperimentConfig, eval_fn; Xo, Δ, state, Ytrue) -> BOResult

Run the Bayesian optimization loop with online noise learning.

At each step:
1. Run RxInfer variational inference (state-space GP + InverseWishart prior on R)
2. Update the R belief: posterior from this step becomes prior for the next
3. Compute UCB acquisition values
4. Select the unobserved point with highest UCB
5. Evaluate the black-box function and update observations
6. Optionally re-tune kernel hyperparameters (controlled by `cfg.tune_every`)

If `cfg.animate` is true, collects plot frames and saves a GIF.
Logs progress every `cfg.log_every` steps via `@info`.
"""
function run_bo!(cfg::ExperimentConfig, eval_fn; Xo, Δ, state::BOState, Ytrue)
    N = cfg.N
    frames = cfg.animate ? [] : nothing

    best_value_history = Float64[]
    n_observed_history = Int[]
    R_diag_history = Vector{Float64}[]

    n_completed = 0
    for step in 1:cfg.steps
        R_init = state.R_prior
        initialization = @initialization begin
            q(R) = R_init
        end
        res = infer(
            model=additive_gp_vv(P=state.blocks.P, A=state.blocks.A, Q=state.blocks.Q, H=state.blocks.H, prior_R=R_init),
            data=(Y=state.Y,),
            iterations=10,
            initialization=initialization,
            predictvars=(Y=KeepLast(),)
        )

        # Update R belief: posterior becomes next step's prior
        state.R_prior = last(res.posteriors[:R])

        acq = ucb_acquisition(res, state, cfg, N)
        k = select_next_point(acq.ucb, state.Y)
        k == 0 && break

        if cfg.animate
            plt = plot_bo_step(step, k, state, acq, Ytrue, cfg)
            push!(frames, plt)
        end

        y_new = eval_fn(row(Xo, k))
        state.Y[k] = (y_new .- state.μy) ./ state.σy

        if cfg.tune_every > 0 && step % cfg.tune_every == 0 && length(findall(!ismissing, state.Y)) >= 5
            params = tune_hyperparameters(state.Y, Δ, state.W, cfg.D, cfg.Q; R_diag_init=diag(mean(state.R_prior))[1])
            state.W = params.W
            state.blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs=params.ℓs, σ2s=params.σ2s, W=params.W)
        end

        n_completed = step

        # Track per-step metrics
        best = _current_best(state, cfg)
        n_obs = length(findall(!ismissing, state.Y))
        R_mean = mean(state.R_prior)
        push!(best_value_history, best.value)
        push!(n_observed_history, n_obs)
        push!(R_diag_history, diag(R_mean))

        if step % cfg.log_every == 0
            R_diag_mean = round.(diag(R_mean); digits=4)
            @info "Step $step/$(cfg.steps)" n_observed=n_obs best_value=round(best.value; digits=4) R_diag=R_diag_mean
        end
    end

    best = _current_best(state, cfg)
    observed = findall(!ismissing, state.Y)
    R_learned = Matrix(mean(state.R_prior))
    result = BOResult(best.index, best.value, best.y, observed, n_completed, R_learned,
                      best_value_history, n_observed_history, R_diag_history)
    (; result, frames)
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
        "best_index"         => result.best_index,
        "best_value"         => result.best_value,
        "best_y"             => result.best_y,
        "n_iterations"       => result.n_iterations,
        "n_observed"         => length(result.observed_indices),
        "R_learned_diag"     => diag(result.R_learned),
        "best_value_history" => result.best_value_history,
        "n_observed_history" => result.n_observed_history,
        "R_diag_history"     => result.R_diag_history,
        "config" => Dict(
            "N" => cfg.N, "d" => cfg.d, "Q" => cfg.Q, "D" => cfg.D,
            "beta" => cfg.β, "n_seed" => cfg.n_seed, "steps" => cfg.steps,
            "tune_every" => cfg.tune_every, "R_diag_init" => cfg.R_diag_init,
            "seed" => cfg.seed, "ls" => cfg.ℓs, "sigma2s" => cfg.σ2s, "s" => cfg.s
        )
    )
    json_path = joinpath(output_dir, "metrics.json")
    open(json_path, "w") do io
        JSON.print(io, metrics, 2)
    end
    @info "Saved metrics to $json_path"

    # PNG of last BO state
    if !isnothing(frames) && !isempty(frames)
        png_path = joinpath(output_dir, "bo_last_state.png")
        savefig(last(frames), png_path)
        @info "Saved last state to $png_path"
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
