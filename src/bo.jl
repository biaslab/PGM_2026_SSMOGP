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

    R = Diagonal(fill(cfg.R_diag_init, cfg.D))

    Y = Vector{Union{Missing, Vector{Float64}}}(undef, cfg.N)
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

    state = BOState(blocks, W, R, Y, μy, σy)
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

Run the Bayesian optimization loop.

At each step:
1. Run RxInfer inference on the state-space GP model
2. Compute UCB acquisition values
3. Select the unobserved point with highest UCB
4. Evaluate the black-box function and update observations
5. Optionally re-tune hyperparameters (controlled by `cfg.tune_every`)

If `cfg.animate` is true, collects plot frames and saves a GIF.
Logs progress every `cfg.log_every` steps via `@info`.
"""
function run_bo!(cfg::ExperimentConfig, eval_fn; Xo, Δ, state::BOState, Ytrue)
    N = cfg.N
    frames = cfg.animate ? [] : nothing

    n_completed = 0
    for step in 1:cfg.steps
        res = infer(
            model=additive_gp_vv(P=state.blocks.P, A=state.blocks.A, Q=state.blocks.Q, H=state.blocks.H, R=state.R),
            data=(Y=state.Y,),
            predictvars=(Y=KeepLast(),)
        )

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
            params = tune_hyperparameters(state.Y, Δ, state.W, cfg.D, cfg.Q; R_diag_init=state.R[1,1])
            state.W = params.W
            state.blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs=params.ℓs, σ2s=params.σ2s, W=params.W)
            state.R = Diagonal(fill(params.R_diag, cfg.D))
        end

        n_completed = step

        if step % cfg.log_every == 0
            best = _current_best(state, cfg)
            n_obs = length(findall(!ismissing, state.Y))
            @info "Step $step/$( cfg.steps)" n_observed=n_obs best_value=round(best.value; digits=4)
        end
    end

    if cfg.animate && !isempty(frames)
        anim = @animate for plt in frames
            plt
        end
        gif(anim, "bo_multioutput_chain.gif", fps=2)
        @info "Animation saved to bo_multioutput_chain.gif"
    end

    best = _current_best(state, cfg)
    observed = findall(!ismissing, state.Y)
    BOResult(best.index, best.value, best.y, observed, n_completed)
end
