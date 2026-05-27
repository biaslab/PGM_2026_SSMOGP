# ETT forecasting under partial (randomly-dropped) observations.
#
# Train a multi-output state-space LMC GP on the first half of the ETTh1 series
# (with random per-feature dropout) and forecast the held-out second half.
# Compares two ways of handling the partial observations:
#   - SS-LMC: reactive message passing (missing entries simply have no factor)
#   - KM-LMC: covariance restructuring — build the full structured (D·N_train)²
#     LMC kernel, slice out the observed (time, feature) sub-block, factorize.
# Reports forecast MNLL, RMSE, and fit+forecast wall-clock time, swept over the
# training-window size N and the dropout level p.

"""
    load_ett(path; n_rows=nothing) -> Matrix{Float64}

Load the numeric ETT columns (drops the leading `date` column) as an
(n_rows × 7) matrix. Parsed manually to avoid a CSV/DataFrames dependency.
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
    _dropout_mask(N_train, N_test, D, p, rng) -> BitMatrix

(N × D) observation mask. Each training-half cell is observed independently with
probability `1 - p`; the entire forecast half is unobserved (held out).
"""
function _dropout_mask(N_train::Int, N_test::Int, D::Int, p::Float64, rng)
    N = N_train + N_test
    mask = falses(N, D)
    for i in 1:N_train, d in 1:D
        mask[i, d] = rand(rng) > p
    end
    mask
end

"""
    _ett_setup(data, N_train, N_test, p, seed, D, Q, ℓs, σ2s) -> NamedTuple

Build everything shared by both forecast methods for one (window, dropout, seed):
the time grid, mixing matrix W, dropout mask, per-feature standardization (from the
training half), point-level `Y` (test half = missing), flat `Y_flat`, and the
original-scale ground truth `Ytrue`.
"""
function _ett_setup(data::AbstractMatrix, N_train::Int, N_test::Int, p::Float64,
                    seed::Int, D::Int, Q::Int)
    N = N_train + N_test
    rng = MersenneTwister(seed)

    W = randn(rng, D, Q) .* 0.5
    mask = _dropout_mask(N_train, N_test, D, p, rng)

    # Per-feature standardization from the training half
    μy = vec(mean(@view(data[1:N_train, :]), dims=1))
    σy = vec(std(@view(data[1:N_train, :]), dims=1)) .+ 1e-8

    Ytrue = [Vector{Float64}(@view data[i, :]) for i in 1:N]

    Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
    fill!(Y, missing)
    Y_flat = Vector{Union{Missing, Float64}}(missing, N * D)
    for i in 1:N_train
        Y[i] = (Ytrue[i] .- μy) ./ σy
        for d in 1:D
            mask[i, d] && (Y_flat[(i-1)*D + d] = Y[i][d])
        end
    end

    (; N, W, mask, μy, σy, Ytrue, Y, Y_flat)
end

_test_rmse(μ_pred, Ytrue, test_idx, D) =
    sqrt(sum(sum(abs2, μ_pred[i] .- Ytrue[i]) for i in test_idx) / (length(test_idx) * D))

# Mean negative log predictive density over the explicit forecast indices.
# (Computed directly here rather than via `_compute_mnll`, whose observed/missing
# selection is tailored to the sequential-design setting.)
function _forecast_mnll(μ_pred, σ_pred, Ytrue, test_idx, D)
    nll = 0.0
    for i in test_idx, d in 1:D
        σ2 = max(σ_pred[i][d]^2, 1e-8)
        nll += 0.5 * log(2π * σ2) + (Ytrue[i][d] - μ_pred[i][d])^2 / (2σ2)
    end
    nll / (length(test_idx) * D)
end

"""
    forecast_ss(setup, N_train, N_test, D, Q, ℓs, σ2s, R_diag_init) -> (; mnll, rmse, time)

SS-LMC forecast via reactive message passing. Missing entries (dropped training
features and the whole forecast half) carry no observation factor; the posterior at
every timestep is read from `res.posteriors[:my]`.
"""
function forecast_ss(setup, N_train::Int, N_test::Int, D::Int, Q::Int,
                     ℓs::Vector{Float64}, σ2s::Vector{Float64}, R_diag_init::Float64)
    N = setup.N
    Xo = reshape(Float64.(1:N), N, 1)
    Δ = ones(N)  # uniform hourly spacing → ℓ in timestep units
    blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs=ℓs, σ2s=σ2s, W=setup.W)
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
    σ_pred = [sqrt.(var(pred[i])) .* setup.σy for i in 1:N]

    test_idx = (N_train + 1):N
    mnll = _forecast_mnll(μ_pred, σ_pred, setup.Ytrue, test_idx, D)
    rmse = _test_rmse(μ_pred, setup.Ytrue, test_idx, D)
    (; mnll, rmse, time=t)
end

"""
    forecast_km(setup, N_train, N_test, D, Q, ℓs, σ2s, R_diag_init) -> (; mnll, rmse, time)

KM-LMC forecast via covariance restructuring. Builds the full structured
(D·N_train)² LMC kernel, slices the observed (time, feature) sub-block, factorizes
(Cholesky), then predicts the forecast half through the cross-covariance to the
observed entries. This is the per-fit cost the paper compares against message passing.
"""
function forecast_km(setup, N_train::Int, N_test::Int, D::Int, Q::Int,
                     ℓs::Vector{Float64}, σ2s::Vector{Float64}, R_diag_init::Float64)
    N = setup.N
    W = setup.W
    mask = setup.mask

    # Observed training (time, feature) pairs → flat indices in train space
    obs_idx = Int[]; obs_dim = Int[]; obs_time = Int[]; y_obs = Float64[]
    for i in 1:N_train, d in 1:D
        if mask[i, d]
            push!(obs_idx, (i-1)*D + d); push!(obs_dim, d); push!(obs_time, i)
            push!(y_obs, setup.Y[i][d])
        end
    end
    M = length(obs_idx)
    selfvar = _lmc_self_variance(W, σ2s)

    local mnll, rmse
    t = @elapsed begin
        Xo_train = reshape(Float64.(1:N_train), N_train, 1)
        # Full structured train kernel, then restructure to the observed sub-block
        Kf = _lmc_full_kernel(Xo_train, W, ℓs, σ2s, 1.0)
        K_obs = Kf[obs_idx, obs_idx] + Diagonal([R_diag_init for _ in obs_dim])
        L = cholesky(Symmetric(K_obs) + 1e-8I)
        α = L \ y_obs

        # Cross-covariance forecast-half × observed pairs (D·N_test × M)
        K_cross = zeros(D * N_test, M)
        for a in 1:M
            io = obs_time[a]; dobs = obs_dim[a]
            for tt in 1:N_test
                r = Float64(abs((N_train + tt) - io))
                for d in 1:D
                    kv = 0.0
                    for q in 1:Q
                        kv += W[d, q] * W[dobs, q] * _matern32_kernel(r, ℓs[q], σ2s[q])
                    end
                    K_cross[(tt-1)*D + d, a] = kv
                end
            end
        end

        μ_flat = K_cross * α
        V = L.L \ K_cross'                       # M × (D·N_test)
        var_flat = [selfvar[((j-1) % D) + 1] for j in 1:(D*N_test)] .- vec(sum(abs2, V; dims=1))

        # Assemble per-timestep predictions (forecast half only) and score
        μ_pred = [setup.μy for _ in 1:N]         # train-half placeholder (unused by metrics)
        σ_pred = [setup.σy for _ in 1:N]
        for tt in 1:N_test
            rng = (tt-1)*D+1 : tt*D
            μ_pred[N_train + tt] = μ_flat[rng] .* setup.σy .+ setup.μy
            σ_pred[N_train + tt] = sqrt.(max.(var_flat[rng], 1e-10)) .* setup.σy
        end

        test_idx = (N_train + 1):N
        mnll = _forecast_mnll(μ_pred, σ_pred, setup.Ytrue, test_idx, D)
        rmse = _test_rmse(μ_pred, setup.Ytrue, test_idx, D)
    end
    (; mnll, rmse, time=t)
end

"""
    run_ett_forecast(data, N, p, seed; D, Q, ℓs, σ2s, R_diag_init) -> (; ss, km)

Run one train(first half)/forecast(second half) comparison at window `N`, dropout
`p`, and `seed`. Returns the (mnll, rmse, time) NamedTuple for each method.
"""
function run_ett_forecast(data, N::Int, p::Float64, seed::Int;
                          D::Int, Q::Int, ℓs::Vector{Float64}, σ2s::Vector{Float64},
                          R_diag_init::Float64)
    N_train = N ÷ 2
    N_test = N - N_train
    setup = _ett_setup(data, N_train, N_test, p, seed, D, Q)
    ss = forecast_ss(setup, N_train, N_test, D, Q, ℓs, σ2s, R_diag_init)
    km = forecast_km(setup, N_train, N_test, D, Q, ℓs, σ2s, R_diag_init)
    (; ss, km)
end

# ─── Sweep runner ────────────────────────────────────────────────────────────

"""
    run_ett_sweeps(data; Ns, ps, N_fixed, p_fixed, seeds, D, Q, ℓs, σ2s,
                   R_diag_init, output_dir) -> (; sweep_N, sweep_p)

Two 1-D sweeps comparing SS-LMC (message passing) vs KM-LMC (cov restructuring):
- sweep over window size `Ns` at fixed dropout `p_fixed`
- sweep over dropout `ps` at fixed window `N_fixed`
Each cell is averaged over `seeds`. Saves comparison JSON and time/MNLL/RMSE plots.
"""
function run_ett_sweeps(data; Ns, ps, N_fixed, p_fixed, seeds,
                        D, Q, ℓs, σ2s, R_diag_init, output_dir="data/partial_obs")
    mkpath(output_dir)

    # Warm up JIT so the first recorded cell isn't inflated by compilation.
    @info "Warming up (compilation)…"
    run_ett_forecast(data, 200, p_fixed, first(seeds);
                     D=D, Q=Q, ℓs=ℓs, σ2s=σ2s, R_diag_init=R_diag_init)

    sweep_N = Dict{String,Any}[]
    for N in Ns
        for seed in seeds
            @info "Sweep-N: N=$N p=$p_fixed seed=$seed"
            r = run_ett_forecast(data, N, p_fixed, seed;
                                  D=D, Q=Q, ℓs=ℓs, σ2s=σ2s, R_diag_init=R_diag_init)
            push!(sweep_N, Dict("N"=>N, "p"=>p_fixed, "seed"=>seed,
                "ss"=>Dict("mnll"=>r.ss.mnll, "rmse"=>r.ss.rmse, "time"=>r.ss.time),
                "km"=>Dict("mnll"=>r.km.mnll, "rmse"=>r.km.rmse, "time"=>r.km.time)))
        end
    end

    sweep_p = Dict{String,Any}[]
    for p in ps
        for seed in seeds
            @info "Sweep-p: N=$N_fixed p=$p seed=$seed"
            r = run_ett_forecast(data, N_fixed, p, seed;
                                  D=D, Q=Q, ℓs=ℓs, σ2s=σ2s, R_diag_init=R_diag_init)
            push!(sweep_p, Dict("N"=>N_fixed, "p"=>p, "seed"=>seed,
                "ss"=>Dict("mnll"=>r.ss.mnll, "rmse"=>r.ss.rmse, "time"=>r.ss.time),
                "km"=>Dict("mnll"=>r.km.mnll, "rmse"=>r.km.rmse, "time"=>r.km.time)))
        end
    end

    open(joinpath(output_dir, "comparison.json"), "w") do io
        JSON.print(io, Dict("sweep_N"=>sweep_N, "sweep_p"=>sweep_p,
                            "p_fixed"=>p_fixed, "N_fixed"=>N_fixed), 2)
    end
    @info "Saved ETT forecast comparison to $(joinpath(output_dir, "comparison.json"))"

    try
        _plot_ett_sweep(sweep_N, "N", Ns, output_dir, p_fixed)
        _plot_ett_sweep(sweep_p, "p", ps, output_dir, N_fixed)
    catch e
        @warn "Plotting failed (results are saved in comparison.json)" exception=e
    end

    (; sweep_N, sweep_p)
end

"""
    _plot_ett_sweep(rows, axis, xs, output_dir, fixed)

Plot time/MNLL/RMSE (mean ± std over seeds) for SS vs KM against the swept axis
(`"N"` or `"p"`). Time is plotted on a log scale.
"""
function _plot_ett_sweep(rows, axis::String, xs, output_dir, fixed)
    _agg(x, m, metric) = begin
        v = [Float64(r[m][metric]) for r in rows if r[axis] == x]
        (mean(v), length(v) > 1 ? std(v) : 0.0)
    end
    xlab = axis == "N" ? "Training window N (p=$fixed)" : "Dropout p (N=$fixed)"

    series = (("ss", "SS-LMC (msg passing)", :blue), ("km", "KM-LMC (cov restruct.)", :red))

    xv = Float64.(collect(xs))
    for (metric, ylab, logy) in (("time", "Fit+forecast time (s)", true),
                                  ("mnll", "Forecast MNLL", false),
                                  ("rmse", "Forecast RMSE", false))
        means = Dict(m => [first(_agg(x, m, metric)) for x in xs] for (m, _, _) in series)
        allm = vcat(values(means)...)
        lo, hi = minimum(allm), maximum(allm)
        pad = max((hi - lo) * 0.5, abs((lo + hi) / 2) * 0.1, 1e-3)
        try
            save_plot(joinpath(output_dir, "$(metric)_vs_$(axis)"); tikz=!logy) do
                kw = logy ? (; yscale=:log10) : (; ylims=(lo - pad, hi + pad))
                p = plot(; xlabel=xlab, ylabel=ylab, legend=:topleft, size=(560, 380),
                         left_margin=8Plots.mm, bottom_margin=6Plots.mm, kw...)
                for (m, lab, col) in series
                    plot!(p, xv, means[m], lw=2, marker=:circle, label=lab, color=col)
                end
                p
            end
        catch e
            @warn "Could not render $(metric)_vs_$(axis)" exception=e
        end
    end
end
