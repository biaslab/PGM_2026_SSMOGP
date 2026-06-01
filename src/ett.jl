# ETT multi-dim-input regression under partial observations.
#
# Splits ETTh1 into a 3D input (the "useful" loads HUFL, MUFL, LUFL) and a 4D
# output (HULL, MULL, LULL, OT). Uses the first N rows as training candidates
# (with random per-feature dropout on outputs) and the next N rows as a fully
# held-out test set. Both halves share the same 3D input space; the SS-LMC
# model handles the multi-dim input via NN-chain ordering. Compares:
#   - SS-LMC: reactive message passing along the NN chain
#   - KM-LMC: full structured LMC kernel + Cholesky (cov restructuring)
#   - SVGP-LMC: variational sparse GP with M inducing points
# Reports forecast MNLL, RMSE, and fit+predict wall-clock time over sweeps in
# the training window C (= N) and dropout p.

"""
    load_ett(path; n_rows=nothing) -> Matrix{Float64}

Load the numeric ETT columns (drops the leading `date` column) as an
(n_rows × 7) matrix. Parsed manually to avoid a CSV/DataFrames dependency.
Column order after dropping `date`:
  1=HUFL, 2=HULL, 3=MUFL, 4=MULL, 5=LUFL, 6=LULL, 7=OT.
"""
function load_ett(path::AbstractString; n_rows::Union{Nothing,Int}=nothing)
    rows = Vector{Vector{Float64}}()
    open(path, "r") do io
        readline(io)  # header
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ',')
            push!(rows, parse.(Float64, parts[2:end]))  # drop date
            n_rows !== nothing && length(rows) >= n_rows && break
        end
    end
    permutedims(reduce(hcat, rows))  # (n_rows × D)
end

"""
    _ett_setup(data, N_train, N_test, p, seed, D, Q, input_cols, output_cols)
        -> NamedTuple

Build everything shared by the three forecast methods for one (window, dropout,
seed). Splits the first `N_train + N_test` rows of `data` into training
candidates (first N_train) and held-out test points (next N_test), both
projected to the chosen `input_cols` (3D) and `output_cols` (4D). Inputs are
standardized by training-half stats and NN-chain-ordered jointly with the test
inputs; per-cell dropout (`p`) is applied only to training-half outputs. The
returned NamedTuple carries `Xo` (chain-ordered inputs), `Δ` (normalized chain
distances), `dist_norm`, dropout `mask`, point-level standardized `Y`, the flat
`Y_flat`, the ground truth `Ytrue`, and `test_idx_chain` — the chain positions
that map back to the original test rows.
"""
function _ett_setup(data::AbstractMatrix, N_train::Int, N_test::Int, p::Float64,
                    seed::Int, D::Int, Q::Int,
                    input_cols::Vector{Int}, output_cols::Vector{Int})
    N = N_train + N_test
    rng = MersenneTwister(seed)

    X_raw = Matrix{Float64}(@view data[1:N, input_cols])    # N × d
    Y_raw = Matrix{Float64}(@view data[1:N, output_cols])   # N × D

    # Input standardization from the training half only.
    μx = vec(mean(@view(X_raw[1:N_train, :]), dims=1))
    σx = vec(std(@view(X_raw[1:N_train, :]), dims=1)) .+ 1e-8
    X = (X_raw .- μx') ./ σx'

    # NN-chain order over all N points (train + test) so they share one chain.
    order = nn_chain_order(X)
    Xo = X[order, :]
    inv_order = invperm(order)                              # original row → chain position

    Δ = zeros(N)
    for i in 2:N
        Δ[i] = sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    dist_norm = median(Δ[2:end])
    Δ ./= dist_norm
    # Floor: very dense chains (small d, large N) produce near-zero consecutive
    # Δ, which makes the SS-LMC Kalman transitions near-singular. The floor is
    # well below typical chain spacing so it only kicks in on degenerate links.
    Δ = max.(Δ, 1e-3)

    W = randn(rng, D, Q) .* 0.5

    # Output standardization from training-half outputs.
    μy = vec(mean(@view(Y_raw[1:N_train, :]), dims=1))
    σy = vec(std(@view(Y_raw[1:N_train, :]), dims=1)) .+ 1e-8

    # Ground truth and dropout mask in chain order.
    Ytrue = [Vector{Float64}(Y_raw[order[i], :]) for i in 1:N]

    mask = falses(N, D)
    for i in 1:N
        original_row = order[i]
        if original_row <= N_train                          # training point
            for d in 1:D
                mask[i, d] = rand(rng) > p
            end
        end
        # test points: mask stays false (outputs fully held out)
    end

    Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
    fill!(Y, missing)
    Y_flat = Vector{Union{Missing, Float64}}(missing, N * D)
    for i in 1:N
        any(mask[i, :]) || continue
        Y[i] = (Ytrue[i] .- μy) ./ σy
        for d in 1:D
            mask[i, d] && (Y_flat[(i-1)*D + d] = Y[i][d])
        end
    end

    test_idx_chain = [inv_order[r] for r in (N_train + 1):N]

    (; N, Xo, Δ, dist_norm, W, mask, μy, σy, Ytrue, Y, Y_flat, test_idx_chain)
end

# Replace NaN/Inf with `nothing` so the result can be serialized as JSON null.
_sanitize_json(x::AbstractFloat) = isfinite(x) ? x : nothing
_sanitize_json(x::Dict) = Dict(k => _sanitize_json(v) for (k, v) in x)
_sanitize_json(x::AbstractVector) = [_sanitize_json(v) for v in x]
_sanitize_json(x) = x

_test_rmse(μ_pred, Ytrue, test_idx, D) =
    sqrt(sum(sum(abs2, μ_pred[i] .- Ytrue[i]) for i in test_idx) / (length(test_idx) * D))

# Mean negative log predictive density over the explicit forecast indices.
function _forecast_mnll(μ_pred, σ_pred, Ytrue, test_idx, D)
    nll = 0.0
    for i in test_idx, d in 1:D
        σ2 = max(σ_pred[i][d]^2, 1e-8)
        nll += 0.5 * log(2π * σ2) + (Ytrue[i][d] - μ_pred[i][d])^2 / (2σ2)
    end
    nll / (length(test_idx) * D)
end

"""
    forecast_ss(setup, D, Q, ℓs, σ2s, R_diag_init) -> (; mnll, rmse, time)

SS-LMC forecast via reactive message passing along the NN chain. Missing
entries (dropped training outputs and all test outputs) carry no observation
factor; predictions are read from `res.posteriors[:my]` at every chain
position, and metrics are scored at `setup.test_idx_chain`.
"""
function forecast_ss(setup, D::Int, Q::Int,
                     ℓs::Vector{Float64}, σ2s::Vector{Float64}, R_diag_init::Float64)
    N = setup.N
    blocks = additive_multioutput_blocks_from_Δ(setup.Δ; ℓs=ℓs, σ2s=σ2s, W=setup.W)
    τ = fill(1.0 / R_diag_init, D)
    e_vecs = [Float64.(I(D)[:, d]) for d in 1:D]

    local res
    t = @elapsed begin
        res = infer(
            model=additive_gp_po(P=blocks.P, A=blocks.A, Q=blocks.Q, H=blocks.H,
                                 τ=τ, e_vecs=e_vecs, N=N, D=D),
            data=(Y=setup.Y_flat,),
            options=(limit_stack_depth=1000,),
        )
    end

    pred = res.posteriors[:my]
    μ_pred = [mean(pred[i]) .* setup.σy .+ setup.μy for i in 1:N]
    σ_pred = [sqrt.(max.(var(pred[i]), 1e-12)) .* setup.σy for i in 1:N]

    mnll = _forecast_mnll(μ_pred, σ_pred, setup.Ytrue, setup.test_idx_chain, D)
    rmse = _test_rmse(μ_pred, setup.Ytrue, setup.test_idx_chain, D)
    (; mnll, rmse, time=t)
end

"""
    forecast_ss_raw(setup, D, Q, ℓs, σ2s, R_diag_init) -> (; mnll, rmse, time)

SS-LMC forecast via a hand-coded Kalman filter + RTS smoother
(`ss_lmc_filter_smooth`, `src/ss_lmc_raw.jl`). Same SS blocks as `forecast_ss`,
but without RxInfer in the loop — used for the fair scalability comparison and
as a numerical diagnostic against the message-passing implementation.
"""
function forecast_ss_raw(setup, D::Int, Q::Int,
                         ℓs::Vector{Float64}, σ2s::Vector{Float64},
                         R_diag_init::Float64)
    N = setup.N
    blocks = additive_multioutput_blocks_from_Δ(setup.Δ; ℓs=ℓs, σ2s=σ2s, W=setup.W)
    τ = fill(1.0 / R_diag_init, D)

    local pred
    t = @elapsed begin
        pred = ss_lmc_filter_smooth(blocks.P, blocks.A, blocks.Q, blocks.H,
                                    τ, setup.Y_flat, N, D)
    end

    μ_pred = [pred.μ_pred[i] .* setup.σy .+ setup.μy for i in 1:N]
    σ_pred = [pred.σ_pred[i] .* setup.σy for i in 1:N]

    mnll = _forecast_mnll(μ_pred, σ_pred, setup.Ytrue, setup.test_idx_chain, D)
    rmse = _test_rmse(μ_pred, setup.Ytrue, setup.test_idx_chain, D)
    (; mnll, rmse, time=t)
end

"""
    forecast_km(setup, D, Q, ℓs, σ2s, R_diag_init) -> (; mnll, rmse, time)

KM-LMC forecast via covariance restructuring on the multi-dim input. Builds
the full structured (D·N)² LMC kernel over the chain-ordered inputs `setup.Xo`,
slices out the observed (chain position, output dim) sub-block, factorizes
(Cholesky), and predicts at every chain position via the cross-covariance.
This is the per-fit cost the paper compares against message passing.
"""
function forecast_km(setup, D::Int, Q::Int,
                     ℓs::Vector{Float64}, σ2s::Vector{Float64}, R_diag_init::Float64)
    N = setup.N
    W = setup.W
    mask = setup.mask

    local mnll, rmse
    t = @elapsed begin
        Kf = _lmc_full_kernel(setup.Xo, W, ℓs, σ2s, setup.dist_norm)

        # Flat indices (point-major) of the observed scalar entries.
        obs_idx = Int[]; obs_dim = Int[]; y_obs = Float64[]
        for i in 1:N
            ismissing(setup.Y[i]) && continue
            for d in 1:D
                if mask[i, d]
                    push!(obs_idx, (i - 1) * D + d)
                    push!(obs_dim, d)
                    push!(y_obs, setup.Y[i][d])
                end
            end
        end

        K_obs = Kf[obs_idx, obs_idx] + Diagonal([R_diag_init for _ in obs_dim])
        L = cholesky(Symmetric(K_obs) + 1e-8I)
        α = L \ y_obs

        cross = Kf[:, obs_idx]                              # (D·N) × M
        μ_flat = cross * α
        V = L.L \ cross'
        var_flat = diag(Kf) .- vec(sum(abs2, V; dims=1))

        μ_pred = Vector{Vector{Float64}}(undef, N)
        σ_pred = Vector{Vector{Float64}}(undef, N)
        for i in 1:N
            rng = (i - 1) * D + 1 : i * D
            μ_pred[i] = μ_flat[rng] .* setup.σy .+ setup.μy
            σ_pred[i] = sqrt.(max.(var_flat[rng], 1e-10)) .* setup.σy
        end

        mnll = _forecast_mnll(μ_pred, σ_pred, setup.Ytrue, setup.test_idx_chain, D)
        rmse = _test_rmse(μ_pred, setup.Ytrue, setup.test_idx_chain, D)
    end
    (; mnll, rmse, time=t)
end

"""
    forecast_svgp(setup, D, Q, ℓs, σ2s, R_diag_init, M; Z_seed=0)
        -> (; mnll, rmse, time)

SVGP-LMC forecast via the closed-form variational posterior in `src/svgp.jl`.
Inducing locations Z are picked by K-means over the training chain positions
(the test inputs are held out of the inducing-point pool). Predictions at all
N chain positions are scored at `setup.test_idx_chain`.
"""
function forecast_svgp(setup, D::Int, Q::Int,
                       ℓs::Vector{Float64}, σ2s::Vector{Float64},
                       R_diag_init::Float64, M::Int; Z_seed::Int=0)
    N = setup.N
    Xo = setup.Xo

    # Training rows in chain order — inducing pool draws from these only.
    train_chain = findall(any.(eachrow(setup.mask)))
    M_use = min(M, max(length(train_chain), 1))

    local mnll, rmse
    t = @elapsed begin
        X_train = isempty(train_chain) ? Xo : Xo[train_chain, :]
        Z = _kmeans_init(X_train, M_use; rng=MersenneTwister(Z_seed))

        state = SVGPState(
            copy(setup.W), copy(ℓs), copy(σ2s),
            fill(R_diag_init, D),
            M_use, Z,
            [zeros(M_use) for _ in 1:Q],
            [Matrix{Float64}(I(M_use)) for _ in 1:Q],
            Matrix{Float64}(I(Q * M_use)),
            zeros(Q * M_use),
            false,
            setup.dist_norm,
            Xo,
            copy(setup.Y), copy(setup.mask),
            copy(setup.μy), copy(setup.σy),
        )

        _fit_svgp!(state)
        pred = _lmc_svgp_predict(state)
        μ_pred = [pred.μ_pred[i] .* setup.σy .+ setup.μy for i in 1:N]
        σ_pred = [pred.σ_pred[i] .* setup.σy for i in 1:N]
        mnll = _forecast_mnll(μ_pred, σ_pred, setup.Ytrue, setup.test_idx_chain, D)
        rmse = _test_rmse(μ_pred, setup.Ytrue, setup.test_idx_chain, D)
    end
    (; mnll, rmse, time=t)
end

"""
    run_ett_forecast(data, N, p, seed; D, Q, ℓs, σ2s, R_diag_init,
                     input_cols, output_cols, M=64) -> (; ss, km, svgp)

Run one train(first N rows)/forecast(next N rows) comparison at window `N`,
dropout `p`, and `seed`. The total data consumed is 2·N rows. Returns the
(mnll, rmse, time) NamedTuple for each method.
"""
function run_ett_forecast(data, N::Int, p::Float64, seed::Int;
                          D::Int, Q::Int, ℓs::Vector{Float64}, σ2s::Vector{Float64},
                          R_diag_init::Float64,
                          input_cols::Vector{Int}, output_cols::Vector{Int},
                          M::Int=64, km_max_N::Int=4000)
    setup = _ett_setup(data, N, N, p, seed, D, Q, input_cols, output_cols)
    ss_raw = forecast_ss_raw(setup, D, Q, ℓs, σ2s, R_diag_init)
    if N <= km_max_N
        km = forecast_km(setup, D, Q, ℓs, σ2s, R_diag_init)
    else
        @info "Skipping KM-LMC at N=$N (km_max_N=$km_max_N): (D·2N)² kernel exceeds memory budget"
        km = (mnll=NaN, rmse=NaN, time=NaN)
    end
    svgp = forecast_svgp(setup, D, Q, ℓs, σ2s, R_diag_init, M; Z_seed=seed)
    (; ss_raw, km, svgp)
end

# ─── Sweep runner ────────────────────────────────────────────────────────────

"""
    run_ett_sweeps(data; Ns, ps, N_fixed, p_fixed, seeds, D, Q,
                   ℓs, σ2s, R_diag_init, input_cols, output_cols, M=64,
                   km_max_N=4000, output_dir) -> (; sweep_N, sweep_p)

Two sweeps comparing SS-LMC (raw Kalman), KM-LMC, and SVGP-LMC under
per-feature dropout on the multi-dim ETT inputs:
- N sweep over `Ns` at fixed `p_fixed`
- p sweep over `ps` at fixed `N_fixed`

KM-LMC is skipped when `N > km_max_N` (its (D·2N)² kernel would exceed memory
at large N). Each cell is averaged over `seeds`. Saves comparison JSON and
time/MNLL/RMSE plots.
"""
function run_ett_sweeps(data; Ns, ps, N_fixed, p_fixed, seeds,
                        D, Q, ℓs, σ2s, R_diag_init,
                        input_cols::Vector{Int}, output_cols::Vector{Int},
                        M::Int=64, km_max_N::Int=4000,
                        output_dir="data/partial_obs")
    mkpath(output_dir)

    # Warm up JIT so the first recorded cell isn't inflated by compilation.
    @info "Warming up (compilation)…"
    run_ett_forecast(data, min(200, first(Ns)), p_fixed, first(seeds);
                     D=D, Q=Q, ℓs=ℓs, σ2s=σ2s, R_diag_init=R_diag_init,
                     input_cols=input_cols, output_cols=output_cols,
                     M=M, km_max_N=km_max_N)

    _row(N, p, seed, r) = Dict(
        "N"     => N, "p"     => p, "seed" => seed,
        "ss_raw"=> Dict("mnll"=>r.ss_raw.mnll,"rmse"=>r.ss_raw.rmse,"time"=>r.ss_raw.time),
        "km"    => Dict("mnll"=>r.km.mnll,    "rmse"=>r.km.rmse,    "time"=>r.km.time),
        "svgp"  => Dict("mnll"=>r.svgp.mnll,  "rmse"=>r.svgp.rmse,  "time"=>r.svgp.time),
    )

    sweep_N = Dict{String,Any}[]
    for N in Ns, seed in seeds
        @info "Sweep-N: N=$N p=$p_fixed seed=$seed"
        r = run_ett_forecast(data, N, p_fixed, seed;
                              D=D, Q=Q, ℓs=ℓs, σ2s=σ2s, R_diag_init=R_diag_init,
                              input_cols=input_cols, output_cols=output_cols,
                              M=M, km_max_N=km_max_N)
        push!(sweep_N, _row(N, p_fixed, seed, r))
    end

    sweep_p = Dict{String,Any}[]
    for p in ps, seed in seeds
        @info "Sweep-p: N=$N_fixed p=$p seed=$seed"
        r = run_ett_forecast(data, N_fixed, p, seed;
                              D=D, Q=Q, ℓs=ℓs, σ2s=σ2s, R_diag_init=R_diag_init,
                              input_cols=input_cols, output_cols=output_cols,
                              M=M, km_max_N=km_max_N)
        push!(sweep_p, _row(N_fixed, p, seed, r))
    end

    payload = Dict("sweep_N"=>sweep_N,
                   "sweep_p"=>sweep_p,
                   "p_fixed"=>p_fixed,
                   "N_fixed"=>N_fixed,
                   "km_max_N"=>km_max_N,
                   "M"=>M)
    open(joinpath(output_dir, "comparison.json"), "w") do io
        JSON.print(io, _sanitize_json(payload), 2)
    end
    @info "Saved ETT forecast comparison to $(joinpath(output_dir, "comparison.json"))"

    try
        _plot_ett_sweep(sweep_N, "N", Ns, output_dir, p_fixed; suffix="")
        _plot_ett_sweep(sweep_p, "p", ps, output_dir, N_fixed; suffix="")
    catch e
        @warn "Plotting failed (results are saved in comparison.json)" exception=e
    end

    (; sweep_N, sweep_p)
end

"""
    _plot_ett_sweep(rows, axis, xs, output_dir, fixed; suffix="")

Plot time/MNLL/RMSE (mean ± std over seeds) for SS, KM, and SVGP against the
swept axis (`"N"` or `"p"`). `suffix` is appended to the output file stem
(e.g. `_at_Cbig`) to distinguish multiple p-sweeps.
"""
function _plot_ett_sweep(rows, axis::String, xs, output_dir, fixed; suffix::String="")
    _agg(x, m, metric) = begin
        v = [Float64(r[m][metric]) for r in rows if r[axis] == x]
        (mean(v), length(v) > 1 ? std(v) : 0.0)
    end
    xlab = axis == "N" ? "Window length C (p=$fixed)" : "Dropout p (C=$fixed)"

    series = (("ss_raw", "SS-LMC (raw Kalman)",       :blue),
              ("km",     "KM-LMC (cov restruct.)",    :red),
              ("svgp",   "SVGP-LMC (M inducing pts)", :green))

    xv = Float64.(collect(xs))
    for (metric, ylab, logy) in (("time", "Fit+forecast time (s)", true),
                                  ("mnll", "Forecast MNLL", false),
                                  ("rmse", "Forecast RMSE", false))
        means = Dict(m => [first(_agg(x, m, metric)) for x in xs] for (m, _, _) in series)
        allm = vcat(values(means)...)
        lo, hi = minimum(allm), maximum(allm)
        pad = max((hi - lo) * 0.5, abs((lo + hi) / 2) * 0.1, 1e-3)
        try
            save_plot(joinpath(output_dir, "$(metric)_vs_$(axis)$(suffix)")) do
                kw = logy ? (; yscale=:log10) : (; ylims=(lo - pad, hi + pad))
                p = plot(; xlabel=xlab, ylabel=ylab, legend=:topleft, size=(560, 380),
                         left_margin=8Plots.mm, bottom_margin=6Plots.mm, kw...)
                for (m, lab, col) in series
                    plot!(p, xv, means[m], lw=2, marker=:circle, label=lab, color=col)
                end
                p
            end
        catch e
            @warn "Could not render $(metric)_vs_$(axis)$(suffix)" exception=e
        end
    end
end
