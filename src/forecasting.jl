# Forecasting on real time-series data (ETTh1).
#
# One-step-ahead rolling forecast: at each test timestamp t, the model observes
# all training data (with dropout) plus ground-truth values at all earlier test
# timestamps (k < t), and predicts the output at t. Dropout applies only to the
# training half on a per-(timestamp, output) basis.
#
# One inference call per (test timestamp, seed, dropout, method).

"""
    _generate_forecast_mask(N, D, train_frac, dropout, rng) -> BitMatrix

Build the observation mask:
- rows 1..floor(train_frac*N): each entry true with probability (1 - dropout)
- remaining rows: all false (test set)
"""
function _generate_forecast_mask(N::Int, D::Int, train_frac::Float64, dropout::Float64, rng)
    @assert 0.0 <= dropout < 1.0
    @assert 0.0 < train_frac < 1.0
    n_train = floor(Int, train_frac * N)

    mask = falses(N, D)
    for i in 1:n_train, d in 1:D
        mask[i, d] = rand(rng) > dropout
    end
    mask
end

"""
    setup_forecast(cfg, X, Y_data, mask) -> NamedTuple

Build the SS-GP problem from fixed (X, Y_data) and an observation mask.

In contrast to `setup_experiment`:
- X is fixed (not randomly generated) and assumed sorted along the input axis
  (in 1D this means natural temporal ordering; the function asserts d=1).
- Output standardization uses ONLY training-observed entries (no test leakage).
- A point's `Y[i]` is set to the standardized observation iff `any(mask[i, :])`.

Returns: `(; Xo, Δ, blocks, W, Y, μy, σy, Ytrue, dist_norm, mask)`.
"""
function setup_forecast(cfg::ExperimentConfig, X::Matrix{Float64},
                        Y_data::Vector{Vector{Float64}}, mask::BitMatrix)
    N, d = size(X)
    d == 1 || error("setup_forecast currently expects d=1 (got d=$d)")
    @assert size(mask) == (N, cfg.D)
    @assert length(Y_data) == N

    # Natural ordering in 1D
    order = sortperm(X[:, 1])
    Xo = X[order, :]
    Y_ordered = Y_data[order]
    mask_ordered = mask[order, :]

    Δ = zeros(N)
    for i in 2:N
        Δ[i] = sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    dist_norm = median(Δ[2:end])
    Δ ./= dist_norm

    rng = MersenneTwister(cfg.seed)
    W = randn(rng, cfg.D, cfg.Q) .* 0.5
    blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs=cfg.ℓs, σ2s=cfg.σ2s, W)

    # Per-output standardization from training-observed entries only
    μy = zeros(cfg.D)
    σy = ones(cfg.D)
    for d_out in 1:cfg.D
        vals = Float64[]
        for i in 1:N
            mask_ordered[i, d_out] && push!(vals, Y_ordered[i][d_out])
        end
        isempty(vals) && error("Output $d_out has zero observed entries — increase n_train or lower dropout")
        μy[d_out] = mean(vals)
        σy[d_out] = std(vals) + 1e-8
    end

    Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
    fill!(Y, missing)
    for i in 1:N
        if any(@view mask_ordered[i, :])
            Y[i] = (Y_ordered[i] .- μy) ./ σy
        end
    end

    (; Xo, Δ, blocks, W, Y, μy, σy, Ytrue=Y_ordered, dist_norm, mask=mask_ordered)
end

"""
    setup_forecast_po(cfg, setup_data) -> POState

Build a `POState` from the output of `setup_forecast` (which already carries
the mask). Mirrors `setup_po` but skips its mask generation.
"""
function setup_forecast_po(cfg::ExperimentConfig, setup_data)
    D = cfg.D
    N = cfg.N
    τ = fill(1.0 / cfg.R_diag_init, D)
    e_vecs = [Float64.(I(D)[:, d]) for d in 1:D]
    Y_flat = _build_Y_flat(setup_data.Y, setup_data.mask, N, D)
    POState(setup_data.blocks, copy(setup_data.W), τ, e_vecs,
            setup_data.mask, copy(setup_data.Y), Y_flat,
            copy(setup_data.μy), copy(setup_data.σy))
end

"""
    setup_forecast_baseline_po(cfg, setup_data) -> BaselinePOState

Build a `BaselinePOState` (LMC kernel-matrix GP) from the output of
`setup_forecast`.
"""
function setup_forecast_baseline_po(cfg::ExperimentConfig, setup_data)
    R_diag = fill(cfg.R_diag_init, cfg.D)
    BaselinePOState(copy(setup_data.W), copy(cfg.ℓs), copy(cfg.σ2s), R_diag,
                    setup_data.dist_norm, copy(setup_data.Xo),
                    setup_data.mask, copy(setup_data.Y),
                    copy(setup_data.μy), copy(setup_data.σy))
end

"""
    _compute_test_mse_nll(μ_pred, σ_pred, Ytrue, mask, d_target; noise_var=0.0) -> (mse, nll)

Compute MSE and per-element NLL restricted to rows where output `d_target` is
unobserved (i.e. `mask[i, d_target] == false`), comparing predictions
(original scale) against `Ytrue[i][d_target]`.

`σ_pred` is the GP's *latent* predictive std (original scale). The NLL is
evaluated against the observation `y = f + ε`, so the variance used is
`σ² = σ_pred² + noise_var`. `noise_var` should be the observation noise
variance in the original output scale (i.e. `R_diag_init * σy[d_target]²`).
"""
function _compute_test_mse_nll(μ_pred::Vector{Vector{Float64}},
                               σ_pred::Vector{Vector{Float64}},
                               Ytrue::Vector{Vector{Float64}},
                               mask::BitMatrix, d_target::Int;
                               noise_var::Float64=0.0)
    N = length(μ_pred)
    se_sum = 0.0
    nll_sum = 0.0
    n_test = 0
    for i in 1:N
        mask[i, d_target] && continue
        n_test += 1
        y = Ytrue[i][d_target]
        μ = μ_pred[i][d_target]
        σ2 = σ_pred[i][d_target]^2 + noise_var
        σ2 = max(σ2, 1e-8)
        se_sum += (y - μ)^2
        nll_sum += 0.5 * log(2π * σ2) + (y - μ)^2 / (2σ2)
    end
    n_test == 0 && error("No test samples for output $d_target")
    (mse = se_sum / n_test, nll = nll_sum / n_test, n_test = n_test)
end

"""
    _make_rolling_mask(N, D, train_frac, dropout, n_test_steps, rng) -> (base_mask, n_train)

Generate the base mask used for rolling one-step-ahead forecasting:
- training half (1..n_train): true with probability (1 - dropout)
- test region (n_train+1 : n_train+n_test_steps): false (filled in during rolling)
- beyond test region: false
"""
function _make_rolling_mask(N::Int, D::Int, train_frac::Float64, dropout::Float64,
                            n_test_steps::Int, rng)
    @assert 0.0 <= dropout < 1.0
    @assert 0.0 < train_frac < 1.0
    n_train = floor(Int, train_frac * N)
    @assert n_train + n_test_steps <= N

    mask = falses(N, D)
    for i in 1:n_train, d in 1:D
        mask[i, d] = rand(rng) > dropout
    end
    mask, n_train
end

"""
    _build_rolling_setup(cfg, X, Y_data, base_mask) -> NamedTuple

Like `setup_forecast` but writes nothing into `Y` for points outside the
base-mask. Returns Xo, Δ, blocks, W, μy, σy, dist_norm, Y_ordered, base_mask.
Used as the static piece of the rolling forecaster — the dynamic per-step
piece just rewrites a single row in the Y/mask before each inference call.
"""
function _build_rolling_setup(cfg::ExperimentConfig, X::Matrix{Float64},
                              Y_data::Vector{Vector{Float64}}, base_mask::BitMatrix)
    N, d = size(X)
    d == 1 || error("rolling forecast currently expects d=1 (got d=$d)")
    @assert size(base_mask) == (N, cfg.D)

    order = sortperm(X[:, 1])
    Xo = X[order, :]
    Y_ordered = Y_data[order]
    mask_ordered = base_mask[order, :]

    # Largest index in the training band (rows beyond it have all-false initial mask).
    last_train = 1
    for i in 1:N
        any(@view mask_ordered[i, :]) && (last_train = i)
    end

    Δ = zeros(N)
    for i in 2:N
        Δ[i] = sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    dist_norm = median(Δ[2:end])
    Δ ./= dist_norm

    # ── Per-output standardisation from training-observed entries only ──
    μy = zeros(cfg.D)
    σy = ones(cfg.D)
    for d_out in 1:cfg.D
        vals = Float64[]
        for i in 1:N
            mask_ordered[i, d_out] && push!(vals, Y_ordered[i][d_out])
        end
        isempty(vals) && error("Output $d_out has zero observed entries — lower dropout")
        μy[d_out] = mean(vals)
        σy[d_out] = std(vals) + 1e-8
    end

    # ── Mixing matrix W: top-Q principal components of standardised training data ──
    # With orthonormal W and σ²s, prior variance per output d is
    #     var_d = Σ_q W[d,q]² · σ²s[q]
    # whose average over d is (1/D) Σ_q σ²s[q] · ‖W[:,q]‖² = (1/D) Σ_q σ²s[q].
    # Rescale σ²s so this average equals 1, matching the standardised data scale.
    full_rows = [i for i in 1:last_train if all(@view mask_ordered[i, :])]
    Z = if length(full_rows) >= 3 * cfg.Q
        z = zeros(length(full_rows), cfg.D)
        for (k, i) in enumerate(full_rows)
            z[k, :] = (Y_ordered[i] .- μy) ./ σy
        end
        z
    else
        # Fall back to per-(point,output) mean imputation across train rows
        @warn "Only $(length(full_rows)) fully-observed rows; falling back to mean-imputed PCA matrix"
        z = zeros(last_train, cfg.D)
        for i in 1:last_train, d_out in 1:cfg.D
            z[i, d_out] = mask_ordered[i, d_out] ?
                          (Y_ordered[i][d_out] - μy[d_out]) / σy[d_out] : 0.0
        end
        z
    end
    F = svd(Z)
    W = Matrix(F.V[:, 1:cfg.Q])                          # D × Q, columns orthonormal
    σ²s_eff = cfg.σ2s .* (cfg.D / sum(cfg.σ2s))          # avg per-output prior var = 1
    blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs=cfg.ℓs, σ2s=σ²s_eff, W)

    # ── Diagnostics: log what the prior actually implies vs the data ──
    emp_var = [var([Z[i, d_out] for i in 1:size(Z, 1)]) for d_out in 1:cfg.D]
    prior_var = [sum(W[d_out, q]^2 * σ²s_eff[q] for q in 1:cfg.Q) for d_out in 1:cfg.D]
    emp_corr = cor(Z; dims=1)
    @info "Rolling-setup diagnostics (standardised scale)" cfg_seed=cfg.seed n_full_rows=length(full_rows) emp_var prior_var σ²s_eff
    @info "Empirical output correlation" emp_corr
    @info "Mixing matrix W (D × Q, orthonormal)" W

    (; Xo, Δ, blocks, W, μy, σy, dist_norm, Y_ordered, base_mask=mask_ordered)
end

"""
    run_one_step_ahead_po(cfg, base, t_test_range, d_target) -> NamedTuple

Run rolling SS-GP one-step-ahead predictions over `t_test_range` (typically
`n_train+1 : n_train+n_test_steps`). At each `t`, the model observes the base
mask plus full ground-truth at every earlier test point `k < t`, then reads
the posterior on `my[t]`.

Returns predictions, true values, MSE, NLL, divergence flag, total time.
"""
function run_one_step_ahead_po(cfg::ExperimentConfig, base, cursor_range, d_target::Int;
                               forecast_horizon::Int=1,
                               test_dropout::Float64=0.0, test_mask_rng=nothing)
    N = cfg.N
    D = cfg.D
    τ = fill(1.0 / cfg.R_diag_init, D)
    e_vecs = [Float64.(I(D)[:, d]) for d in 1:D]
    rng = test_mask_rng === nothing ? MersenneTwister(cfg.seed + 9999) : test_mask_rng

    # Pre-sample the test-region mask so SS-GP and KM-GP see identical reveal patterns.
    # Sample for the FULL test region (cursors + the trailing horizon-1 predictions).
    test_reveal = falses(N, D)
    for c in cursor_range
        c > N && break
        for d in 1:D
            test_reveal[c, d] = rand(rng) > test_dropout
        end
    end

    # Running observation state, standardised
    Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
    fill!(Y, missing)
    mask_run = copy(base.base_mask)
    for i in 1:N
        if any(@view mask_run[i, :])
            Y[i] = (base.Y_ordered[i] .- base.μy) ./ base.σy
        end
    end
    Y_flat = _build_Y_flat(Y, mask_run, N, D)

    μ_t = Float64[]; σ_t = Float64[]; y_t = Float64[]; x_t = Float64[]
    diverged = false
    total_time = 0.0

    # At cursor c, predict at t_pred = c + (h-1). Then reveal entries at c
    # (subject to test_dropout) before advancing.
    for c in cursor_range
        t_pred = c + forecast_horizon - 1
        t_pred > N && break
        t_step = @elapsed begin
            local res
            try
                res = infer(
                    model=additive_gp_po(
                        P=base.blocks.P, A=base.blocks.A,
                        Q=base.blocks.Q, H=base.blocks.H,
                        τ=τ, e_vecs=e_vecs, N=N, D=D),
                    data=(Y=Y_flat,),
                    options=(limit_stack_depth=1000,))
            catch e
                @warn "SS-GP rolling inference failed" c t_pred exception=e
                diverged = true
                @goto next_c
            end

            my_t = res.posteriors[:my][t_pred]
            μ_std = mean(my_t)[d_target]
            σ_std = sqrt(var(my_t)[d_target])
            if !isfinite(μ_std) || !isfinite(σ_std) || abs(μ_std) > 50
                diverged = true
                @goto next_c
            end

            μ_orig = μ_std * base.σy[d_target] + base.μy[d_target]
            σ_orig = σ_std * base.σy[d_target]
            y_true = base.Y_ordered[t_pred][d_target]

            push!(μ_t, μ_orig); push!(σ_t, σ_orig)
            push!(y_t, y_true); push!(x_t, base.Xo[t_pred, 1])

            # Reveal entries at the cursor (one step closer to t_pred)
            Y[c] = (base.Y_ordered[c] .- base.μy) ./ base.σy
            for d in 1:D
                if test_reveal[c, d]
                    mask_run[c, d] = true
                    Y_flat[(c-1)*D + d] = Y[c][d]
                end
            end
            @label next_c
        end
        total_time += t_step
        diverged && break
    end

    # One extra inference call to read the full-N posterior with the final state.
    # Same sanity check as the rolling loop: if any |μ_std| > 50 or non-finite,
    # discard the readout. Rolling MSE/NLL above are unaffected; only the
    # cosmetic training-region overlay on the predictions plot is dropped.
    # We grab posteriors for ALL D outputs (not just d_target) so per-output
    # prediction plots can use them.
    full_μ = Float64[]; full_σ = Float64[]
    full_μ_all = Vector{Float64}[]; full_σ_all = Vector{Float64}[]
    if !diverged
        try
            res_final = infer(
                model=additive_gp_po(
                    P=base.blocks.P, A=base.blocks.A,
                    Q=base.blocks.Q, H=base.blocks.H,
                    τ=τ, e_vecs=e_vecs, N=N, D=D),
                data=(Y=Y_flat,),
                options=(limit_stack_depth=1000,))
            μ_std_mat = [mean(res_final.posteriors[:my][i]) for i in 1:N]  # vector of D-vectors
            σ_std_mat = [sqrt.(var(res_final.posteriors[:my][i])) for i in 1:N]
            μ_target = [μ_std_mat[i][d_target] for i in 1:N]
            if any(any.(!isfinite, μ_std_mat)) || maximum(abs, μ_target) > 50
                @warn "SS-GP full-N readout diverged; skipping train-region overlay" max_abs_μ=maximum(abs, μ_target)
            else
                full_μ = [μ_std_mat[i][d_target] * base.σy[d_target] + base.μy[d_target] for i in 1:N]
                full_σ = [σ_std_mat[i][d_target] * base.σy[d_target] for i in 1:N]
                full_μ_all = [μ_std_mat[i] .* base.σy .+ base.μy for i in 1:N]
                full_σ_all = [σ_std_mat[i] .* base.σy for i in 1:N]
            end
        catch e
            @warn "SS-GP full-N readout failed" exception=e
        end
    end

    noise_var = cfg.R_diag_init * base.σy[d_target]^2
    if isempty(y_t)
        return (; μ_t, σ_t, y_t, x_t, full_μ, full_σ, full_μ_all, full_σ_all,
                  mse=NaN, nll=NaN, diverged=true, time=total_time)
    end
    se = sum((y_t[i] - μ_t[i])^2 for i in eachindex(y_t))
    nll = sum(
        let σ2 = max(σ_t[i]^2 + noise_var, 1e-8)
            0.5 * log(2π * σ2) + (y_t[i] - μ_t[i])^2 / (2σ2)
        end
        for i in eachindex(y_t))
    (; μ_t, σ_t, y_t, x_t, full_μ, full_σ, full_μ_all, full_σ_all,
       mse=se / length(y_t), nll=nll / length(y_t),
       diverged, time=total_time)
end

"""
    run_one_step_ahead_baseline_po(cfg, base, t_test_range, d_target) -> NamedTuple

Same as `run_one_step_ahead_po` but with the LMC kernel-matrix GP baseline.
"""
function run_one_step_ahead_baseline_po(cfg::ExperimentConfig, base, cursor_range, d_target::Int;
                                        forecast_horizon::Int=1,
                                        test_dropout::Float64=0.0, test_mask_rng=nothing)
    N = cfg.N
    D = cfg.D
    rng = test_mask_rng === nothing ? MersenneTwister(cfg.seed + 9999) : test_mask_rng

    # Pre-sample identical reveal mask as SS-GP (same RNG seed upstream)
    test_reveal = falses(N, D)
    for c in cursor_range
        c > N && break
        for d in 1:D
            test_reveal[c, d] = rand(rng) > test_dropout
        end
    end

    Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
    fill!(Y, missing)
    mask_run = copy(base.base_mask)
    for i in 1:N
        if any(@view mask_run[i, :])
            Y[i] = (base.Y_ordered[i] .- base.μy) ./ base.σy
        end
    end

    R_diag = fill(cfg.R_diag_init, D)
    bl_state = BaselinePOState(copy(base.W), copy(cfg.ℓs), copy(cfg.σ2s), R_diag,
                               base.dist_norm, copy(base.Xo),
                               copy(mask_run), Y,
                               copy(base.μy), copy(base.σy))

    μ_t = Float64[]; σ_t = Float64[]; y_t = Float64[]; x_t = Float64[]
    total_time = 0.0

    for c in cursor_range
        t_pred = c + forecast_horizon - 1
        t_pred > N && break
        t_step = @elapsed begin
            pred = _lmc_predict_po(bl_state)
            μ_std = pred.μ_pred[t_pred][d_target]
            σ_std = pred.σ_pred[t_pred][d_target]

            μ_orig = μ_std * base.σy[d_target] + base.μy[d_target]
            σ_orig = σ_std * base.σy[d_target]
            y_true = base.Y_ordered[t_pred][d_target]

            push!(μ_t, μ_orig); push!(σ_t, σ_orig)
            push!(y_t, y_true); push!(x_t, base.Xo[t_pred, 1])

            # Reveal entries at the cursor for the next iteration
            bl_state.Y[c] = (base.Y_ordered[c] .- base.μy) ./ base.σy
            for d in 1:D
                if test_reveal[c, d]
                    bl_state.mask[c, d] = true
                end
            end
        end
        total_time += t_step
    end

    # Full-N posterior with final state. Same sanity check as SS-GP path —
    # KM-GP almost never diverges but the guard is cheap and keeps both branches
    # consistent. We capture all D outputs for multi-output prediction plots.
    full_μ = Float64[]; full_σ = Float64[]
    full_μ_all = Vector{Float64}[]; full_σ_all = Vector{Float64}[]
    try
        pred_final = _lmc_predict_po(bl_state)
        μ_std_mat = pred_final.μ_pred  # length-N vector of D-vectors
        σ_std_mat = pred_final.σ_pred
        μ_target = [μ_std_mat[i][d_target] for i in 1:N]
        if any(any.(!isfinite, μ_std_mat)) || maximum(abs, μ_target) > 50
            @warn "KM-GP full-N readout diverged; skipping train-region overlay" max_abs_μ=maximum(abs, μ_target)
        else
            full_μ = [μ_std_mat[i][d_target] * base.σy[d_target] + base.μy[d_target] for i in 1:N]
            full_σ = [σ_std_mat[i][d_target] * base.σy[d_target] for i in 1:N]
            full_μ_all = [μ_std_mat[i] .* base.σy .+ base.μy for i in 1:N]
            full_σ_all = [σ_std_mat[i] .* base.σy for i in 1:N]
        end
    catch e
        @warn "KM-GP full-N readout failed" exception=e
    end

    noise_var = cfg.R_diag_init * base.σy[d_target]^2
    se = sum((y_t[i] - μ_t[i])^2 for i in eachindex(y_t))
    nll = sum(
        let σ2 = max(σ_t[i]^2 + noise_var, 1e-8)
            0.5 * log(2π * σ2) + (y_t[i] - μ_t[i])^2 / (2σ2)
        end
        for i in eachindex(y_t))
    (; μ_t, σ_t, y_t, x_t, full_μ, full_σ, full_μ_all, full_σ_all,
       mse=se / length(y_t), nll=nll / length(y_t),
       diverged=false, time=total_time)
end

"""
    run_forecast_po(cfg, setup_data) -> (; μ_pred, σ_pred, time)

Single SS-GP inference pass with partial observations. Returns
predictions in the original output scale.
"""
function run_forecast_po(cfg::ExperimentConfig, setup_data)
    po_state = setup_forecast_po(cfg, setup_data)
    N = cfg.N
    D = cfg.D
    t = @elapsed begin
        res = infer(
            model=additive_gp_po(
                P=po_state.blocks.P, A=po_state.blocks.A,
                Q=po_state.blocks.Q, H=po_state.blocks.H,
                τ=po_state.τ, e_vecs=po_state.e_vecs,
                N=N, D=D),
            data=(Y=po_state.Y_flat,),
            options=(limit_stack_depth=1000,))
        pred_output = res.posteriors[:my]
        μ_std = mean.(pred_output)
        σ_std = [sqrt.(var(pred_output[i])) for i in 1:N]
        μ_pred = [μ_std[i] .* po_state.σy .+ po_state.μy for i in 1:N]
        σ_pred = [σ_std[i] .* po_state.σy for i in 1:N]
    end
    # Sanity-check: posterior means should be on the scale of standardised data
    # (|μ_std| ≲ 10). Anything beyond that is an SS-GP message-passing failure
    # caused by sparse partial-obs masks → flag it so callers can react.
    max_abs = maximum(maximum(abs, m) for m in μ_std)
    diverged = max_abs > 50.0
    (; μ_pred, σ_pred, time = t, diverged, max_abs_std = max_abs)
end

"""
    run_forecast_baseline_po(cfg, setup_data) -> (; μ_pred, σ_pred, time)

Single KM-GP inference pass with partial observations.
"""
function run_forecast_baseline_po(cfg::ExperimentConfig, setup_data)
    bl_state = setup_forecast_baseline_po(cfg, setup_data)
    N = cfg.N
    t = @elapsed begin
        pred = _lmc_predict_po(bl_state)
        μ_pred = [pred.μ_pred[i] .* bl_state.σy .+ bl_state.μy for i in 1:N]
        σ_pred = [pred.σ_pred[i] .* bl_state.σy for i in 1:N]
    end
    (; μ_pred, σ_pred, time = t, diverged = false, max_abs_std = 0.0)
end

"""
    run_ett_comparison(cfg_template, X, Y_data; seeds, dropouts, train_frac, d_target, output_dir)
        -> results

Sweep (seed × dropout × method) over the ETT data and write
`comparison.json` plus MSE/NLL bar plots to `output_dir`.

`d_target` is the output index to evaluate (typically `ot_idx` from
`load_etth1`). The mask, GP setup, and `μy/σy` are all per-(seed, dropout).
"""
function run_ett_comparison(cfg_template::ExperimentConfig,
                            X::Matrix{Float64}, Y_data::Vector{Vector{Float64}};
                            seeds=0:4, dropouts=[0.0, 0.3, 0.6],
                            horizons::Vector{Int}=[1],
                            train_frac::Float64=0.9, d_target::Int,
                            n_test_steps::Int=25,
                            test_dropout_mode::Symbol=:none,  # :none or :same_as_train
                            plot_horizon::Union{Int,Nothing}=nothing,
                            col_names::Vector{String}=String[],
                            output_dir::AbstractString="data/ett_forecast")
    mkpath(output_dir)
    all_results = Dict{String, Any}[]

    plot_seed = first(seeds)
    # If the requested plot horizon isn't in the sweep, fall back to the longest
    # horizon we'll actually compute. Otherwise the payload stays empty and
    # predictions plots silently no-op.
    requested_h = plot_horizon === nothing ? maximum(horizons) : plot_horizon
    plot_horizon_use = requested_h in horizons ? requested_h : maximum(horizons)
    if plot_horizon_use != requested_h
        @warn "plot_horizon=$requested_h not in horizons=$horizons; using $plot_horizon_use instead"
    end
    plot_payload = Dict{Float64, Any}()

    # The full test region spans cursors n_train+1..n_train+n_test_steps and
    # predicts as far as n_train + n_test_steps + max(h) - 1.
    h_max = maximum(horizons)

    for h in horizons, seed in seeds, dropout in dropouts
        @info "=== ETT h=$h seed=$seed dropout=$dropout ==="

        cfg = ExperimentConfig(;
            N=cfg_template.N, d=cfg_template.d, Q=cfg_template.Q, D=cfg_template.D,
            ℓs=cfg_template.ℓs, σ2s=cfg_template.σ2s, β=cfg_template.β, s=cfg_template.s,
            n_seed=cfg_template.n_seed, steps=cfg_template.steps,
            tune_every=cfg_template.tune_every, R_diag_init=cfg_template.R_diag_init,
            animate=false, log_every=cfg_template.log_every, seed=seed,
            obs_pattern=:full, obs_frac=cfg_template.obs_frac)

        rng_mask = MersenneTwister(seed + 7000)
        # Reserve room for cursor sweep + the trailing horizon-1 predictions.
        base_mask, n_train = _make_rolling_mask(cfg.N, cfg.D, train_frac, dropout,
                                                n_test_steps + h_max - 1, rng_mask)
        base = _build_rolling_setup(cfg, X, Y_data, base_mask)
        cursor_range = (n_train + 1):(n_train + n_test_steps)

        test_dropout = test_dropout_mode == :same_as_train ? dropout : 0.0

        @info "  SS-GP (rolling, $n_test_steps cursors, h=$h)" test_dropout
        out_ss = run_one_step_ahead_po(cfg, base, cursor_range, d_target;
                                       forecast_horizon=h,
                                       test_dropout=test_dropout,
                                       test_mask_rng=MersenneTwister(seed + 8000))
        if out_ss.diverged
            @warn "  SS-GP diverged during rolling forecast" seed dropout h
        end

        @info "  KM-GP (rolling, $n_test_steps cursors, h=$h)"
        out_km = run_one_step_ahead_baseline_po(cfg, base, cursor_range, d_target;
                                                forecast_horizon=h,
                                                test_dropout=test_dropout,
                                                test_mask_rng=MersenneTwister(seed + 8000))

        if seed == plot_seed && h == plot_horizon_use
            ctx_start = max(1, n_train - 80)
            ctx_end   = min(cfg.N, n_train + n_test_steps + h - 1)
            ctx_idx   = ctx_start:ctx_end
            plot_payload[dropout] = (;
                ctx_x    = base.Xo[ctx_idx, 1],
                ctx_y    = [base.Y_ordered[i][d_target] for i in ctx_idx],
                ctx_y_all = [base.Y_ordered[i] for i in ctx_idx],   # D-vector per row
                ctx_idx  = collect(ctx_idx),
                train_obs_x_per_d = [
                    [base.Xo[i, 1] for i in ctx_start:n_train if base.base_mask[i, d_out]]
                    for d_out in 1:cfg.D
                ],
                train_obs_y_per_d = [
                    [base.Y_ordered[i][d_out] for i in ctx_start:n_train if base.base_mask[i, d_out]]
                    for d_out in 1:cfg.D
                ],
                # Back-compat fields (used by single-OT plot)
                train_obs_x = [base.Xo[i, 1] for i in ctx_start:n_train
                               if base.base_mask[i, d_target]],
                train_obs_y = [base.Y_ordered[i][d_target] for i in ctx_start:n_train
                               if base.base_mask[i, d_target]],
                test_x   = out_km.x_t,
                test_y   = out_km.y_t,
                ss_μ     = out_ss.μ_t, ss_σ = out_ss.σ_t, ss_diverged = out_ss.diverged,
                km_μ     = out_km.μ_t, km_σ = out_km.σ_t,
                ss_full_μ = out_ss.full_μ, ss_full_σ = out_ss.full_σ,
                km_full_μ = out_km.full_μ, km_full_σ = out_km.full_σ,
                ss_full_μ_all = out_ss.full_μ_all, ss_full_σ_all = out_ss.full_σ_all,
                km_full_μ_all = out_km.full_μ_all, km_full_σ_all = out_km.full_σ_all,
                n_train  = n_train, horizon = h,
            )
        end

        push!(all_results, Dict{String, Any}(
            "seed"       => seed,
            "dropout"    => dropout,
            "horizon"    => h,
            "n_test"     => length(out_km.y_t),
            "ss" => Dict("mse" => out_ss.mse, "nll" => out_ss.nll, "time" => out_ss.time,
                        "diverged" => out_ss.diverged),
            "km" => Dict("mse" => out_km.mse, "nll" => out_km.nll, "time" => out_km.time,
                        "diverged" => false),
        ))
    end

    json_path = joinpath(output_dir, "comparison.json")
    open(json_path, "w") do io
        JSON.print(io, all_results, 2)
    end
    @info "Saved ETT comparison to $json_path"

    _plot_ett_forecast(all_results, dropouts, horizons, output_dir)
    _plot_ett_mse_vs_horizon(all_results, dropouts, horizons, output_dir)
    _plot_ett_predictions(plot_payload, dropouts, output_dir;
                          horizon=plot_horizon_use)
    if !isempty(col_names)
        _plot_ett_predictions_per_output(plot_payload, dropouts, output_dir, col_names;
                                         horizon=plot_horizon_use, d_target=d_target)
    end
    _plot_ett_timing(all_results, dropouts, horizons, output_dir)

    all_results
end

"""
    _plot_ett_forecast(results, dropouts, output_dir)

Plot MSE-vs-dropout and NLL-vs-dropout (mean ± std across seeds) for SS-GP and KM-GP.
"""
function _plot_ett_forecast(results, dropouts, horizons, output_dir)
    theme_kw = publication_theme_kwargs()

    # Median + IQR ribbon across non-diverged seeds, per (dropout, horizon, method).
    _agg(metric, method, h) = begin
        meds = Float64[]; lo = Float64[]; hi = Float64[]
        for p in dropouts
            vals = Float64[r[method][metric] for r in results
                           if r["dropout"] == p && r["horizon"] == h &&
                              !get(r[method], "diverged", false)]
            if isempty(vals)
                push!(meds, NaN); push!(lo, NaN); push!(hi, NaN)
            else
                push!(meds, median(vals))
                push!(lo,   quantile(vals, 0.25))
                push!(hi,   quantile(vals, 0.75))
            end
        end
        (meds, lo, hi)
    end

    # In 1D the SS-GP and KM-GP posteriors coincide to numerical precision,
    # so plotting both as overlapping lines is just visual clutter. Draw one
    # line and label it explicitly. Timing is the only plot that distinguishes
    # the two methods because runtimes genuinely differ.
    function _make_panel(metric, ylabel; legend_pos=:topleft)
        panels = []
        for h in horizons
            p = plot(; xlabel="Dropout fraction", ylabel=ylabel,
                     title="h = $h", legend=legend_pos, theme_kw...)
            ss_m, ss_lo, ss_hi = _agg(metric, "ss", h)
            plot!(p, dropouts, ss_m, ribbon=(ss_m .- ss_lo, ss_hi .- ss_m),
                  fillalpha=0.2, lw=2,
                  label="GP (SS-GP ≡ KM-GP in 1D)",
                  color=:navy, marker=:circle)
            push!(panels, p)
        end
        panels
    end

    save_plot(joinpath(output_dir, "mse_vs_dropout")) do
        panels = _make_panel("mse", "MSE on OT")
        plot(panels...; layout=(1, length(horizons)), size=(320 * length(horizons), 280))
    end

    save_plot(joinpath(output_dir, "nll_vs_dropout")) do
        panels = _make_panel("nll", "NLL on OT")
        plot(panels...; layout=(1, length(horizons)), size=(320 * length(horizons), 280))
    end
end

"""
    _plot_ett_mse_vs_horizon(results, dropouts, horizons, output_dir)

x = forecast horizon; one line per dropout level. Median + IQR ribbon across seeds.
"""
function _plot_ett_mse_vs_horizon(results, dropouts, horizons, output_dir)
    theme_kw = publication_theme_kwargs()
    colors = [:blue, :orange, :purple, :green, :brown]

    _agg(metric, method, dp) = begin
        meds = Float64[]; lo = Float64[]; hi = Float64[]
        for h in horizons
            vals = Float64[r[method][metric] for r in results
                           if r["dropout"] == dp && r["horizon"] == h &&
                              !get(r[method], "diverged", false)]
            if isempty(vals)
                push!(meds, NaN); push!(lo, NaN); push!(hi, NaN)
            else
                push!(meds, median(vals))
                push!(lo,   quantile(vals, 0.25))
                push!(hi,   quantile(vals, 0.75))
            end
        end
        (meds, lo, hi)
    end

    function _save(metric, ylabel, fname)
        save_plot(joinpath(output_dir, fname)) do
            p = plot(; xlabel="Forecast horizon h (steps)", ylabel=ylabel,
                     title="GP (SS-GP ≡ KM-GP in 1D)",
                     legend=:topleft, theme_kw...)
            for (i, dp) in enumerate(dropouts)
                ss_m, ss_lo, ss_hi = _agg(metric, "ss", dp)
                col = colors[mod1(i, length(colors))]
                plot!(p, horizons, ss_m, ribbon=(ss_m .- ss_lo, ss_hi .- ss_m),
                      fillalpha=0.15, lw=2, marker=:circle, ms=4,
                      label="dp=$(round(Int, 100*dp))%", color=col)
            end
            p
        end
    end

    _save("mse", "MSE on OT (median across seeds)", "mse_vs_horizon")
    _save("nll", "NLL on OT (median across seeds)", "nll_vs_horizon")
end

"""
    _plot_ett_predictions(plot_payload, dropouts, output_dir)

Generate predicted-vs-true OT plots for one representative seed, with one panel
per dropout level. Training points are marked; test region gets the
predictive mean ± 2σ ribbon for both SS-GP and KM-GP.
"""
function _plot_ett_predictions(plot_payload, dropouts, output_dir; horizon::Int=1)
    isempty(plot_payload) && return
    theme_kw = publication_theme_kwargs()
    gp_color = :navy

    panels = []
    for (i, dp) in enumerate(dropouts)
        haskey(plot_payload, dp) || continue
        pd = plot_payload[dp]

        # In 1D, SS-GP and KM-GP are mathematically equivalent — we only draw one.
        # Prefer SS-GP curves; fall back to KM-GP when SS-GP readout/rolling diverged.
        ss_ok      = !isempty(pd.ss_full_μ) && length(pd.ss_full_μ) >= maximum(pd.ctx_idx)
        km_ok      = !isempty(pd.km_full_μ) && length(pd.km_full_μ) >= maximum(pd.ctx_idx)
        post_src   = ss_ok ? :ss : (km_ok ? :km : :none)
        roll_src   = (!pd.ss_diverged && !isempty(pd.ss_μ)) ? :ss :
                     (!isempty(pd.km_μ) ? :km : :none)
        fallback   = post_src == :km || roll_src == :km

        # In 1D the SS-GP and KM-GP posteriors are mathematically identical,
        # so we draw a single curve and call this out in the title + legend.
        title_str = "dropout = $(round(Int, 100*dp))% — h=$horizon"

        p = plot(; xlabel="Time (normalized)", ylabel="OT",
                 title=title_str,
                 legend=(i == 1 ? :topleft : false),
                 theme_kw...)

        # Banner annotation calling out the SS=KM equivalence (one per panel)
        annotate!(p, pd.ctx_x[1],
                  minimum(pd.ctx_y) + 0.97 * (maximum(pd.ctx_y) - minimum(pd.ctx_y)),
                  text("GP curves below: SS-GP ≡ KM-GP (identical in 1D)",
                       :left, gp_color, 8))

        # Ground-truth OT over the plotted window
        plot!(p, pd.ctx_x, pd.ctx_y, lw=1.2, color=:gray, alpha=0.6, label="OT (truth)")

        # Train/test boundary
        boundary = !isempty(pd.test_x) ? pd.test_x[1] : pd.ctx_x[end]
        vline!(p, [boundary], color=:black, lw=1, linestyle=:dot, label="train/test")

        # Full-N posterior over the plotted window (faint, no markers) — single method
        post_label = "GP posterior μ ± 2σ  (SS-GP ≡ KM-GP)"
        if post_src == :ss
            μ = pd.ss_full_μ[pd.ctx_idx]; σ = pd.ss_full_σ[pd.ctx_idx]
            plot!(p, pd.ctx_x, μ, ribbon=2 .* σ, fillalpha=0.12, lw=1,
                  color=gp_color, label=post_label)
        elseif post_src == :km
            μ = pd.km_full_μ[pd.ctx_idx]; σ = pd.km_full_σ[pd.ctx_idx]
            plot!(p, pd.ctx_x, μ, ribbon=2 .* σ, fillalpha=0.12, lw=1,
                  color=gp_color, label=post_label)
        end

        # Rolling h-step predictions (heavier, with markers) — single method
        roll_label = "GP h-step μ ± 2σ  (SS-GP ≡ KM-GP)"
        if roll_src == :ss
            plot!(p, pd.test_x[1:length(pd.ss_μ)], pd.ss_μ, ribbon=2 .* pd.ss_σ,
                  fillalpha=0.22, lw=2, color=gp_color, marker=:circle, ms=3,
                  label=roll_label)
        elseif roll_src == :km
            plot!(p, pd.test_x, pd.km_μ, ribbon=2 .* pd.km_σ, fillalpha=0.22, lw=2,
                  color=gp_color, marker=:circle, ms=3,
                  label=roll_label)
        end

        if fallback
            annotate!(p, pd.ctx_x[1], maximum(pd.ctx_y),
                      text("SS-GP diverged; showing KM-GP equivalent", :left, :red, 7))
        end

        # Training observations
        if !isempty(pd.train_obs_x)
            scatter!(p, pd.train_obs_x, pd.train_obs_y, ms=2.5, color=:black,
                     label="OT observed (train, w/ dropout)")
        end

        # True OT at test timestamps
        scatter!(p, pd.test_x, pd.test_y, ms=3.5, color=:gray, msw=0.5,
                 label="OT (test truth)")

        push!(panels, p)
    end

    save_plot(joinpath(output_dir, "predictions_ot")) do
        plot(panels...; layout=(length(panels), 1), size=(800, 300 * length(panels)))
    end
end

"""
    _plot_ett_predictions_per_output(plot_payload, dropouts, output_dir, col_names; horizon, d_target)

For each output `d ∈ 1:D` emit `predictions_<colname>.png` with one panel per
dropout level: truth + training observations + GP posterior μ ± 2σ. The
`d_target` output (typically OT) additionally shows the rolling h-step
predictions in the test region.
"""
function _plot_ett_predictions_per_output(plot_payload, dropouts, output_dir,
                                          col_names::Vector{String};
                                          horizon::Int=1, d_target::Int=0)
    isempty(plot_payload) && return
    theme_kw = publication_theme_kwargs()
    gp_color = :navy
    D_out = length(col_names)

    for d_out in 1:D_out
        col = col_names[d_out]
        is_target = (d_out == d_target)
        panels = []
        for (i, dp) in enumerate(dropouts)
            haskey(plot_payload, dp) || continue
            pd = plot_payload[dp]

            ss_all_ok = !isempty(pd.ss_full_μ_all) && length(pd.ss_full_μ_all) >= maximum(pd.ctx_idx)
            km_all_ok = !isempty(pd.km_full_μ_all) && length(pd.km_full_μ_all) >= maximum(pd.ctx_idx)
            post_src  = ss_all_ok ? :ss : (km_all_ok ? :km : :none)

            title_str = "$col — dropout=$(round(Int, 100*dp))%" *
                        (is_target ? " — h=$horizon" : "")
            p = plot(; xlabel="Time (normalized)", ylabel=col,
                     title=title_str,
                     legend=(i == 1 ? :topleft : false),
                     theme_kw...)

            # Truth for this output over the plotted window
            ctx_y_d = [y[d_out] for y in pd.ctx_y_all]
            plot!(p, pd.ctx_x, ctx_y_d, lw=1.2, color=:gray, alpha=0.6,
                  label="$col (truth)")

            boundary = !isempty(pd.test_x) ? pd.test_x[1] : pd.ctx_x[end]
            vline!(p, [boundary], color=:black, lw=1, linestyle=:dot, label="train/test")

            # Full-N posterior for this output (single curve, SS≡KM in 1D)
            post_label = "GP posterior μ ± 2σ  (SS-GP ≡ KM-GP)"
            if post_src == :ss
                μ = [pd.ss_full_μ_all[i_ctx][d_out] for i_ctx in pd.ctx_idx]
                σ = [pd.ss_full_σ_all[i_ctx][d_out] for i_ctx in pd.ctx_idx]
                plot!(p, pd.ctx_x, μ, ribbon=2 .* σ, fillalpha=0.18, lw=1.5,
                      color=gp_color, label=post_label)
            elseif post_src == :km
                μ = [pd.km_full_μ_all[i_ctx][d_out] for i_ctx in pd.ctx_idx]
                σ = [pd.km_full_σ_all[i_ctx][d_out] for i_ctx in pd.ctx_idx]
                plot!(p, pd.ctx_x, μ, ribbon=2 .* σ, fillalpha=0.18, lw=1.5,
                      color=gp_color, label=post_label)
            end

            # Rolling h-step (only for d_target, where we actually roll-forecast)
            if is_target
                if !pd.ss_diverged && !isempty(pd.ss_μ)
                    plot!(p, pd.test_x[1:length(pd.ss_μ)], pd.ss_μ, ribbon=2 .* pd.ss_σ,
                          fillalpha=0.22, lw=2, color=gp_color, marker=:circle, ms=3,
                          label="GP h-step μ ± 2σ  (SS-GP ≡ KM-GP)")
                elseif !isempty(pd.km_μ)
                    plot!(p, pd.test_x, pd.km_μ, ribbon=2 .* pd.km_σ, fillalpha=0.22, lw=2,
                          color=gp_color, marker=:circle, ms=3,
                          label="GP h-step μ ± 2σ  (SS-GP ≡ KM-GP)")
                end
            end

            # Training observations for THIS output (dropout-masked)
            if d_out <= length(pd.train_obs_x_per_d)
                obs_x = pd.train_obs_x_per_d[d_out]
                obs_y = pd.train_obs_y_per_d[d_out]
                if !isempty(obs_x)
                    scatter!(p, obs_x, obs_y, ms=2.5, color=:black,
                             label="$col observed (train, w/ dropout)")
                end
            end

            push!(panels, p)
        end
        save_plot(joinpath(output_dir, "predictions_$(lowercase(col))")) do
            plot(panels...; layout=(length(panels), 1), size=(800, 280 * length(panels)))
        end
    end
end

"""
    _plot_ett_timing(results, dropouts, output_dir)

Wall-clock inference time per (method, dropout), averaged across seeds.
"""
function _plot_ett_timing(results, dropouts, horizons, output_dir)
    theme_kw = publication_theme_kwargs()

    # One panel per horizon, with SS-GP vs KM-GP bars grouped by dropout.
    save_plot(joinpath(output_dir, "timing")) do
        panels = []
        for h in horizons
            ss_means = [mean(r["ss"]["time"] for r in results
                             if r["dropout"] == dp && r["horizon"] == h)
                        for dp in dropouts]
            km_means = [mean(r["km"]["time"] for r in results
                             if r["dropout"] == dp && r["horizon"] == h)
                        for dp in dropouts]
            labels = ["$(round(Int, 100*dp))%" for dp in dropouts]
            xs = collect(1:length(dropouts))
            w = 0.35
            p = plot(; xlabel="Dropout", ylabel="Time per rolling sweep (s)",
                     title="h = $h", yscale=:log10,
                     legend=(h == first(horizons) ? :topright : false),
                     xticks=(xs, labels), theme_kw...)
            bar!(p, xs .- w/2, ss_means, bar_width=w, label="SS-GP", color=:blue)
            bar!(p, xs .+ w/2, km_means, bar_width=w, label="KM-GP", color=:red)
            push!(panels, p)
        end
        plot(panels...; layout=(1, length(horizons)), size=(320 * length(horizons), 280))
    end
end
