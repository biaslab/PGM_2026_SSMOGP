# Chain starting-point sensitivity vs input dimension (Experiment B: sweep M).
#
# Proposition 1 bounds the starting-point effect in terms of the candidate chain
# length C, but the paper's central tradeoff is that chain quality degrades as
# input dimension M grows (the chain must compress higher-dimensional geometry
# into a 1D Markov sequence). This experiment tests the natural conjecture that
# starting-point SENSITIVITY grows with M: on the synthetic sensor-network
# benchmark (where M can be swept, unlike the fixed M=3 ETT data) we measure the
# spread of held-out SS-LMC RMSE/MNLL across chain starts as a function of M.
#
# Only the ordering varies across the starts of a given seed: the candidate set X
# and mixing matrix W are drawn once per seed (start-independent), and the
# train/test split is fixed on PHYSICAL points and then mapped into each chain
# (so every start observes the same points, isolating path dependence).
#
# Paper notation: M input dim, D outputs, L latents, C candidate chain length,
# gamma2 output scales, Delta chain stretch, T(pi) chain path length, eps_pi
# chain-distance distortion.
#
# Run via: julia --project=. experiments/chain_start_dim.jl
# Or via:  dvc repro chain_start_dim

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using YAML, JSON, Statistics, Random, Plots

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

# ─── Shared helpers (mirrors chain_start_sensitivity.jl) ─────────────────────
function nn_chain_order_from(X::AbstractMatrix{<:Real}, start::Int)
    N = size(X, 1)
    remaining = collect(1:N)
    deleteat!(remaining, findfirst(==(start), remaining))
    order = Vector{Int}(undef, N)
    order[1] = start
    for i in 2:N
        last = order[i-1]
        j = argmin([sqdist(row(X, k), row(X, last)) for k in remaining])
        order[i] = remaining[j]
        splice!(remaining, j)
    end
    order
end

function chain_distortion(Xo::AbstractMatrix{<:Real})
    N = size(Xo, 1)
    cum = zeros(N)
    for i in 2:N
        cum[i] = cum[i-1] + sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    eps = 0.0
    @inbounds for i in 1:N-1
        for j in i+1:N
            de = abs(sqrt(sqdist(row(Xo, i), row(Xo, j))) - (cum[j] - cum[i]))
            de > eps && (eps = de)
        end
    end
    (; eps_pi = eps, T_pi = cum[N])
end

function _edge_jaccard(orders::Vector{Vector{Int}})
    edgeset(o) = Set((min(o[i],o[i+1]), max(o[i],o[i+1])) for i in eachindex(o)[1:end-1])
    es = [edgeset(o) for o in orders]
    ov = Float64[]
    for a in 1:length(es), b in a+1:length(es)
        push!(ov, length(intersect(es[a], es[b])) / length(union(es[a], es[b])))
    end
    isempty(ov) ? NaN : mean(ov)
end

_finite(v) = filter(isfinite, v)
_med(v)    = (f = _finite(v); isempty(f) ? NaN : median(f))
function _stats(v::Vector{Float64})
    f = _finite(v)
    isempty(f) && return Dict("mean"=>nothing,"std"=>nothing,"min"=>nothing,
                              "max"=>nothing,"cv"=>nothing,"n"=>0)
    m = mean(f); s = length(f) > 1 ? std(f) : 0.0
    Dict("mean"=>m,"std"=>s,"min"=>minimum(f),"max"=>maximum(f),
         "cv"=> m != 0 ? s/abs(m) : nothing,"n"=>length(f))
end

# Held-out metrics (mirrors dim_sweep.jl `_compute_rmse_test` / `_compute_mnll_obs`;
# MNLL includes the observation-noise floor R·σy² so it is well-posed where the
# smoother's latent variance underflows at densely-chained test points).
function _rmse_test(μ, Yt, test_idx, D)
    s = 0.0
    for i in test_idx, d in 1:D
        s += (μ[i][d] - Yt[i][d])^2
    end
    sqrt(s / (length(test_idx) * D))
end
function _mnll_obs(μ, σ, Yt, test_idx, σy, R, D)
    nll = 0.0
    for i in test_idx, d in 1:D
        v = σ[i][d]^2 + R * σy[d]^2
        nll += 0.5 * log(2π * v) + (Yt[i][d] - μ[i][d])^2 / (2v)
    end
    nll / (length(test_idx) * D)
end

# ─── Setup builder: sensor network, start-parameterized chain, physical split ─
# Mirrors `setup_experiment` (src/bo.jl) + `_train_test_split` (src/dim_sweep.jl)
# but orders the chain from `start` and fixes the train/test split on physical
# points. Runs SS-LMC via the raw Kalman filter + RTS smoother.
function run_ss_start(eval_fn, N::Int, M::Int, D::Int, L::Int,
                      ℓs::Vector{Float64}, γ2s::Vector{Float64}, R::Float64,
                      train_frac::Float64, seed::Int, start::Int)
    rng = MersenneTwister(seed)
    X = randn(rng, N, M)
    μx = vec(mean(X, dims=1)); σx = vec(std(X, dims=1)) .+ eps()
    X = (X .- μx') ./ σx'

    order = nn_chain_order_from(X, start)
    Xo = X[order, :]
    inv_order = invperm(order)

    Δ = zeros(N)
    for i in 2:N
        Δ[i] = sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    dist_norm = median(Δ[2:end])
    Δ ./= dist_norm

    W = randn(rng, D, L) .* 0.5   # drawn after X (order uses no rng) → fixed across starts

    Ytrue = [eval_fn(row(Xo, i)) for i in 1:N]

    # Train/test split fixed on PHYSICAL points, then mapped into the chain.
    n_train = round(Int, train_frac * N)
    perm = randperm(MersenneTwister(seed + 7777), N)
    train_idx = sort([inv_order[r] for r in perm[1:n_train]])
    test_idx  = sort([inv_order[r] for r in perm[n_train+1:end]])

    Y_train_mat = hcat([Ytrue[k] for k in train_idx]...)'
    μy = vec(mean(Y_train_mat, dims=1)); σy = vec(std(Y_train_mat, dims=1)) .+ 1e-8

    blocks = additive_multioutput_blocks_from_Δ(Δ; ℓs=ℓs, σ2s=γ2s, W=W)
    τ = fill(1.0 / R, D)

    Y_flat = Vector{Union{Missing, Float64}}(missing, N * D)
    for i in train_idx
        ys = (Ytrue[i] .- μy) ./ σy
        for d in 1:D
            Y_flat[(i-1)*D + d] = ys[d]
        end
    end

    pred = ss_lmc_filter_smooth(blocks.P, blocks.A, blocks.Q, blocks.H, τ, Y_flat, N, D)
    μ_pred = [pred.μ_pred[i] .* σy .+ μy for i in 1:N]
    σ_pred = [pred.σ_pred[i] .* σy for i in 1:N]

    rmse = _rmse_test(μ_pred, Ytrue, test_idx, D)
    mnll = _mnll_obs(μ_pred, σ_pred, Ytrue, test_idx, σy, R, D)
    (; rmse, mnll, Xo, order, dist_norm)
end

# ─── Plots (paper export style) ──────────────────────────────────────────────
# Spread across starts (std of per-start-median metric) + chain stretch, vs M.
function _std_vs_M_plot(summary, ds, output_dir)
    save_plot(joinpath(output_dir, "std_vs_M")) do
        Ms = Float64.(ds)
        panels = []
        for (key, ylab) in (("rmse", "RMSE std across starts"),
                            ("mnll", "MNLL std across starts"),
                            ("delta", "Mean chain stretch Δ"))
            p = plot(; xlabel="Input dimension M", ylabel=ylab,
                     xscale=:log2, xticks=(Ms, string.(ds)), legend=false,
                     left_margin=8Plots.mm, bottom_margin=6Plots.mm)
            ys = Float64[let s = summary["M=$d"]
                    key == "delta" ? s["mean_delta_mean"] :
                        (s["ss_$(key)"]["std"] === nothing ? NaN : s["ss_$(key)"]["std"])
                end for d in ds]
            plot!(p, Ms, ys; color=:blue, lw=2, marker=:circle)
            push!(panels, p)
        end
        plot(panels...; layout=(1, 3), size=(960, 340))
    end
end

# Per-start median RMSE cloud vs M (shows the spread widening).
function _spread_vs_M_plot(perstart, ds, output_dir)
    save_plot(joinpath(output_dir, "rmse_spread_vs_M")) do
        jr = MersenneTwister(0)
        p = plot(; xlabel="Input dimension M", ylabel="SS-LMC held-out RMSE",
                 xscale=:log2, xticks=(Float64.(ds), string.(ds)), legend=false,
                 left_margin=8Plots.mm, bottom_margin=6Plots.mm)
        for d in ds
            vals = _finite(Float64[r["ss_rmse"] for r in perstart[d]])
            isempty(vals) && continue
            jx = [d * (2.0 ^ (0.05 * (rand(jr) - 0.5))) for _ in vals]
            scatter!(p, jx, vals; color=:blue, markersize=4, markerstrokewidth=0, alpha=0.5)
            plot!(p, [d], [mean(vals)]; yerror=[std(vals)], color=:black,
                  marker=:diamond, markersize=5)
        end
        p
    end
end

# ─── Main ────────────────────────────────────────────────────────────────────
params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
pc = params["chain_start_dim"]

ds          = Int.(pc["ds"])
N           = Int(pc["N"])              # candidate chain length C
D           = Int(pc["D"])
L           = Int(pc["Q"])              # latents (paper L)
ℓs          = Float64.(pc["ls"])
γ2s         = Float64.(pc["sigma2s"])   # output scales γ²
R_diag_init = Float64(pc["R_diag_init"])
train_frac  = Float64(get(pc, "train_frac", 0.5))
n_starts    = Int(pc["n_starts"])
seeds       = Int.(pc["seeds"])

output_dir = joinpath(@__DIR__, "..", "data", "chain_start_dim")
mkpath(output_dir)

@info "Warming up (compilation)…"
let ef = make_sensor_network(; d=first(ds), D=D)
    run_ss_start(ef, 200, first(ds), D, L, ℓs, γ2s, R_diag_init, train_frac, first(seeds), 1)
end

records = Dict{String,Any}[]           # per (M, start, seed)
orders_by_M = Dict{Int,Vector{Vector{Int}}}()

for M in ds
    eval_fn = make_sensor_network(; d=M, D=D)
    starts = unique(round.(Int, range(1, N, length=n_starts)))
    orders_by_M[M] = Vector{Int}[]
    @info "Chain-start × M sweep" M=M C=N n_starts=length(starts)
    for start in starts
        chain_done = false
        for seed in seeds
            r = run_ss_start(eval_fn, N, M, D, L, ℓs, γ2s, R_diag_init,
                             train_frac, seed, start)
            if !chain_done
                cd = chain_distortion(r.Xo)
                cq = nn_chain_quality(r.Xo)
                push!(orders_by_M[M], r.order)
                push!(records, Dict("M"=>M, "start"=>start, "seed"=>seed,
                    "ss_rmse"=>r.rmse, "ss_mnll"=>r.mnll,
                    "T_pi"=>cd.T_pi, "eps_pi"=>cd.eps_pi,
                    "mean_delta"=>cq.mean_delta, "max_delta"=>cq.max_delta))
                chain_done = true
            else
                push!(records, Dict("M"=>M, "start"=>start, "seed"=>seed,
                    "ss_rmse"=>r.rmse, "ss_mnll"=>r.mnll,
                    "T_pi"=>NaN, "eps_pi"=>NaN, "mean_delta"=>NaN, "max_delta"=>NaN))
            end
        end
    end
end

# ─── Per-(M,start) medians over seeds ────────────────────────────────────────
perstart = Dict{Int,Vector{Dict{String,Any}}}()
for M in ds
    starts = unique(round.(Int, range(1, N, length=n_starts)))
    rows = Dict{String,Any}[]
    for start in starts
        sub = [r for r in records if r["M"] == M && r["start"] == start]
        chain = first(r for r in sub if isfinite(r["T_pi"]))
        push!(rows, Dict("start"=>start,
            "ss_rmse"=>_med(Float64[r["ss_rmse"] for r in sub]),
            "ss_mnll"=>_med(Float64[r["ss_mnll"] for r in sub]),
            "T_pi"=>chain["T_pi"], "eps_pi"=>chain["eps_pi"],
            "mean_delta"=>chain["mean_delta"], "max_delta"=>chain["max_delta"]))
    end
    perstart[M] = rows
end

# ─── Save ────────────────────────────────────────────────────────────────────
_safe(x) = (x isa AbstractFloat && !isfinite(x)) ? nothing : x
_safe(d::Dict) = Dict(k => _safe(v) for (k, v) in d)
_safe(v::AbstractVector) = [_safe(x) for x in v]

open(joinpath(output_dir, "comparison.json"), "w") do io
    JSON.print(io, Dict("ds"=>ds, "C"=>N, "seeds"=>seeds, "n_starts"=>n_starts,
                        "records"=>_safe(records)), 2)
end
@info "Saved raw per-(start,seed) records to comparison.json"

summary = Dict{String,Any}()
for M in ds
    rows = perstart[M]
    col(k) = Float64[r[k] for r in rows]
    summary["M=$M"] = Dict(
        "n_starts"=>length(rows),
        "ss_rmse"=>_stats(col("ss_rmse")), "ss_mnll"=>_stats(col("ss_mnll")),
        "T_pi"=>_stats(col("T_pi")), "eps_pi"=>_stats(col("eps_pi")),
        "mean_delta_mean"=>mean(_finite(col("mean_delta"))),
        "edge_jaccard"=>_edge_jaccard(orders_by_M[M]),
    )
end

metrics = Dict("ds"=>ds, "C"=>N, "n_seeds"=>length(seeds),
               "n_starts"=>n_starts, "per_M"=>summary)
open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.print(io, _safe(metrics), 2)
end
@info "Saved per-M variance summary to metrics.json"

try
    _std_vs_M_plot(summary, ds, output_dir)
    _spread_vs_M_plot(perstart, ds, output_dir)
catch e
    @warn "Plotting failed (results are saved in JSON)" exception=e
end

@info "Synthetic chain starting-point sensitivity vs M (Experiment B) complete"
for M in ds
    s = summary["M=$M"]
    @info "M=$M" ss_rmse_mean=s["ss_rmse"]["mean"] ss_rmse_std=s["ss_rmse"]["std"] ss_mnll_std=s["ss_mnll"]["std"] mean_delta=s["mean_delta_mean"] edge_jaccard=s["edge_jaccard"]
end
