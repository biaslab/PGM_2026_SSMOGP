# Chain starting-point sensitivity on ETT (Experiment A: sweep over C).
#
# Empirically characterizes how the SS-LMC forecast RMSE/MNLL vary when the
# greedy NN-chain is grown from different starting points, on real ETTh1 data.
# This is the empirical companion to Proposition 1: two greedy chains from
# different starts have chain lengths within an O(log C) factor, so the
# posterior-error bound of Theorem 1 varies by at most O(log C) across starts.
# We measure the ACTUAL spread of held-out RMSE/MNLL vs the candidate chain
# length C (= 2N here: N training + N held-out rows ordered jointly).
#
# KM-LMC is included as an order-invariant control: its kernel is
# permutation-invariant (residual start-dependence only through dist_norm), so a
# near-flat KM band next to the SS band shows the SS spread is genuine path
# dependence. Dropout is fixed on PHYSICAL rows per seed and the mixing matrix W
# is fixed per seed, so across the starts of a given seed only the ORDERING
# changes. We aggregate per start as the median over seeds, then report the
# spread across starts.
#
# Paper notation: M input dim, D outputs, L latents, C candidate chain length,
# gamma2 output scales, Delta chain stretch, T(pi) chain path length, eps_pi
# chain-distance distortion.
#
# Run via: julia --project=. experiments/chain_start_sensitivity.jl
# Or via:  dvc repro chain_start_sensitivity

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using YAML, JSON, Statistics, Random, Plots

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

# ─── Start-parameterized greedy NN-chain ordering ────────────────────────────
# Identical to `RxBayesOpt.nn_chain_order` except the first chain point is `start`
# instead of row 1. With `start == 1` it reproduces the shipped ordering exactly.
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

# ε_π: maximum distortion between Euclidean and chain (arc-length) distance over
# all pairs (Theorem 1's quantity). T(π) = total path length = cum[end]. Both are
# seed-independent (depend only on the ordering), so computed once per (N, start).
function chain_distortion(Xo::AbstractMatrix{<:Real})
    N = size(Xo, 1)
    cum = zeros(N)
    for i in 2:N
        cum[i] = cum[i-1] + sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    eps = 0.0
    @inbounds for i in 1:N-1
        for j in i+1:N
            e = sqrt(sqdist(row(Xo, i), row(Xo, j)))
            de = abs(e - (cum[j] - cum[i]))
            de > eps && (eps = de)
        end
    end
    (; eps_pi = eps, T_pi = cum[N])
end

# ─── Setup builder with physical-row dropout ─────────────────────────────────
# Mirrors `RxBayesOpt._ett_setup` but (a) orders the chain from `start`, and
# (b) draws the dropout mask over physical rows with a fixed `seed` before
# reordering into the chain — so only the ordering varies across starts of a
# given seed. Returns the NamedTuple `_ett_setup` produces plus `order`, so the
# unmodified `forecast_ss_raw`/`forecast_km` consume it directly.
function build_setup_start(data::AbstractMatrix, N_train::Int, N_test::Int, p::Float64,
                           seed::Int, start::Int, D::Int, L::Int,
                           input_cols::Vector{Int}, output_cols::Vector{Int})
    N = N_train + N_test
    rng = MersenneTwister(seed)

    X_raw = Matrix{Float64}(@view data[1:N, input_cols])
    Y_raw = Matrix{Float64}(@view data[1:N, output_cols])

    μx = vec(mean(@view(X_raw[1:N_train, :]), dims=1))
    σx = vec(std(@view(X_raw[1:N_train, :]), dims=1)) .+ 1e-8
    X = (X_raw .- μx') ./ σx'

    order = nn_chain_order_from(X, start)
    Xo = X[order, :]
    inv_order = invperm(order)

    Δ = zeros(N)
    for i in 2:N
        Δ[i] = sqrt(sqdist(row(Xo, i), row(Xo, i - 1)))
    end
    dist_norm = median(Δ[2:end])
    Δ ./= dist_norm
    Δ = max.(Δ, 1e-3)

    W = randn(rng, D, L) .* 0.5   # fixed per seed (independent of start)

    μy = vec(mean(@view(Y_raw[1:N_train, :]), dims=1))
    σy = vec(std(@view(Y_raw[1:N_train, :]), dims=1)) .+ 1e-8

    Ytrue = [Vector{Float64}(Y_raw[order[i], :]) for i in 1:N]

    # Physical-row dropout: draw once over training rows, map into chain space.
    mask_phys = falses(N_train, D)
    for r in 1:N_train, d in 1:D
        mask_phys[r, d] = rand(rng) > p
    end
    mask = falses(N, D)
    for i in 1:N
        orig = order[i]
        orig <= N_train || continue
        for d in 1:D
            mask[i, d] = mask_phys[orig, d]
        end
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

    (; N, Xo, Δ, dist_norm, W, mask, μy, σy, Ytrue, Y, Y_flat, test_idx_chain, order)
end

# ─── Aggregation helpers ─────────────────────────────────────────────────────
_finite(v) = filter(isfinite, v)
_med(v)    = (f = _finite(v); isempty(f) ? NaN : median(f))

function _stats(v::Vector{Float64})
    f = _finite(v)
    isempty(f) && return Dict("mean"=>nothing,"std"=>nothing,"min"=>nothing,
                              "max"=>nothing,"cv"=>nothing,"iqr"=>nothing,"n"=>0)
    m = mean(f); s = length(f) > 1 ? std(f) : 0.0
    q = length(f) > 1 ? (quantile(f, 0.75) - quantile(f, 0.25)) : 0.0
    Dict("mean"=>m,"std"=>s,"min"=>minimum(f),"max"=>maximum(f),
         "cv"=> m != 0 ? s/abs(m) : nothing,"iqr"=>q,"n"=>length(f))
end

# Bootstrap 95% CI for Pearson correlation by resampling the paired points.
function _boot_cor_ci(xs::Vector{Float64}, ys::Vector{Float64}; nboot::Int=2000, seed::Int=0)
    keep = isfinite.(xs) .& isfinite.(ys)
    x = xs[keep]; y = ys[keep]
    n = length(x)
    (n < 4 || std(x) == 0 || std(y) == 0) && return (; r=nothing, lo=nothing, hi=nothing)
    rng = MersenneTwister(seed)
    rs = Float64[]
    for _ in 1:nboot
        idx = rand(rng, 1:n, n)
        xb = x[idx]; yb = y[idx]
        (std(xb) == 0 || std(yb) == 0) && continue
        push!(rs, cor(xb, yb))
    end
    isempty(rs) && return (; r=cor(x,y), lo=nothing, hi=nothing)
    (; r=cor(x, y), lo=quantile(rs, 0.025), hi=quantile(rs, 0.975))
end

# Mean pairwise undirected edge-Jaccard across a set of chain orderings — a
# diversity measure (1.0 = identical chains, 0.0 = no shared adjacencies).
function _edge_jaccard(orders::Vector{Vector{Int}})
    edgeset(o) = Set((min(o[i],o[i+1]), max(o[i],o[i+1])) for i in 1:length(o)-1)
    es = [edgeset(o) for o in orders]
    ov = Float64[]
    for a in 1:length(es), b in a+1:length(es)
        push!(ov, length(intersect(es[a], es[b])) / length(union(es[a], es[b])))
    end
    isempty(ov) ? NaN : mean(ov)
end

# ─── Start-selection schemes ─────────────────────────────────────────────────
# Standardized joint input matrix (C × M) for a window N — used to pick starts.
function _ett_inputs(data, N, input_cols)
    C = 2N
    Xr = Matrix{Float64}(@view data[1:C, input_cols])
    μx = vec(mean(@view(Xr[1:N, :]), dims=1)); σx = vec(std(@view(Xr[1:N, :]), dims=1)) .+ 1e-8
    (Xr .- μx') ./ σx'
end

_even_starts(C, k) = unique(round.(Int, range(1, C, length=k)))
_random_starts(C, k, seed) = sort(unique(rand(MersenneTwister(seed), 1:C, 3k)))[1:k]
# Farthest-point sampling of starts: maximally spread starting points (an
# adversarial attempt to force diverse chains).
function _farthest_starts(X, k)
    C = size(X, 1)
    c = vec(mean(X, dims=1))
    S = [argmax([sqdist(row(X, i), c) for i in 1:C])]
    while length(S) < k
        best = -1.0; bi = 0
        for i in 1:C
            i in S && continue
            md = minimum(sqdist(row(X, i), row(X, s)) for s in S)
            md > best && (best = md; bi = i)
        end
        push!(S, bi)
    end
    S
end

# ─── Plots (paper export style: ~560x380 + tikz via save_plot) ───────────────
# Per-C zoomed panels of the per-start median metric; SS vs order-invariant KM.
function _spread_plot(perstart, Ns, ss_key, km_key, ylab, fname, output_dir)
    save_plot(joinpath(output_dir, fname)) do
        panels = []
        for (idx, N) in enumerate(Ns)
            C = 2N
            rows = perstart[N]
            starts = Float64[r["start"] for r in rows]
            sp = plot(; title="C=$C", xlabel="chain start (row index)",
                      ylabel = idx == 1 ? ylab : "",
                      legend = idx == 1 ? :best : false,
                      titlefontsize=9, left_margin=8Plots.mm, bottom_margin=8Plots.mm)
            for (mkey, col, lab) in ((ss_key, :blue, "SS-LMC"),
                                     (km_key, :red,  "KM-LMC (order-inv.)"))
                v = Float64[r[mkey] for r in rows]
                keep = isfinite.(v)
                any(keep) || continue
                scatter!(sp, starts[keep], v[keep]; color=col, markersize=4,
                         markerstrokewidth=0, label=lab)
                hline!(sp, [mean(v[keep])]; color=col, ls=:dash, label="")
            end
            push!(panels, sp)
        end
        plot(panels...; layout=(1, length(Ns)), size=(340 * length(Ns), 360))
    end
end

# Spread across starts (std) vs candidate chain length C, SS vs KM.
function _std_vs_C_plot(summary, Ns, output_dir)
    save_plot(joinpath(output_dir, "std_vs_C")) do
        Cs = Float64[2N for N in Ns]
        panels = []
        for (metric, ylab) in (("rmse", "RMSE std across starts"),
                               ("mnll", "MNLL std across starts"))
            p = plot(; xlabel="Candidate chain length C", ylabel=ylab,
                     xscale=:log10, xticks=(Cs, string.(Int.(Cs))),
                     legend=:best, left_margin=8Plots.mm, bottom_margin=6Plots.mm)
            for (m, col, lab) in (("ss", :blue, "SS-LMC"), ("km", :red, "KM-LMC"))
                ys = Float64[summary["C=$(2N)"]["$(m)_$(metric)"]["std"] === nothing ? NaN :
                             summary["C=$(2N)"]["$(m)_$(metric)"]["std"] for N in Ns]
                plot!(p, Cs, ys; color=col, lw=2, marker=:circle, label=lab)
            end
            push!(panels, p)
        end
        plot(panels...; layout=(1, 2), size=(760, 360))
    end
end

# Per-start median metric vs chain distortion ε_π, one series per C.
function _distortion_plot(perstart, Ns, output_dir)
    save_plot(joinpath(output_dir, "metric_vs_distortion")) do
        panels = []
        for (mkey, ylab) in (("ss_rmse", "SS-LMC RMSE"), ("ss_mnll", "SS-LMC MNLL"))
            p = plot(; xlabel="Chain distortion ε_π", ylabel=ylab,
                     legend=:best, left_margin=8Plots.mm, bottom_margin=6Plots.mm)
            for N in Ns
                rows = perstart[N]
                xs = Float64[r["eps_pi"] for r in rows]
                ys = Float64[r[mkey] for r in rows]
                keep = isfinite.(xs) .& isfinite.(ys)
                any(keep) || continue
                scatter!(p, xs[keep], ys[keep]; label="C=$(2N)",
                         markersize=4, markerstrokewidth=0)
            end
            push!(panels, p)
        end
        plot(panels...; layout=(1, 2), size=(760, 360))
    end
end

# Start-scheme robustness: RMSE std across starts and chain diversity per scheme.
function _scheme_plot(scheme_summary, Ns, output_dir)
    save_plot(joinpath(output_dir, "scheme_comparison")) do
        Cs = Float64[2N for N in Ns]
        panels = []
        for (metric, ylab) in (("rmse_std", "RMSE std across starts"),
                               ("edge_jaccard", "Chain edge-Jaccard (↓ = more diverse)"))
            pp = plot(; xlabel="Candidate chain length C", ylabel=ylab, xscale=:log10,
                      xticks=(Cs, string.(Int.(Cs))), legend=:best,
                      left_margin=8Plots.mm, bottom_margin=6Plots.mm)
            for (name, col) in (("even", :blue), ("random", :orange), ("diverse", :green))
                ys = Float64[scheme_summary["C=$(2N)/$name"][metric] for N in Ns]
                plot!(pp, Cs, ys; color=col, lw=2, marker=:circle, label=name)
            end
            push!(panels, pp)
        end
        plot(panels...; layout=(1, 2), size=(760, 360))
    end
end

# ─── Main ────────────────────────────────────────────────────────────────────
params = YAML.load_file(joinpath(@__DIR__, "..", "params.yaml"))
pc = params["chain_start_sensitivity"]

data_path   = joinpath(@__DIR__, "..", pc["data_path"])
Ns          = Int.(pc["Ns"])
n_starts    = Int(pc["n_starts"])
p_drop      = Float64(pc["p"])
seeds       = Int.(pc["seeds"])
D           = Int(pc["D"])
L           = Int(pc["L"])
ℓs          = Float64.(pc["ls"])
γ2s         = Float64.(pc["gamma2s"])
R_diag_init = Float64(pc["R_diag_init"])
km_max_N    = Int(get(pc, "km_max_N", 2000))
input_cols  = Int.(pc["input_cols"])
output_cols = Int.(pc["output_cols"])

max_rows = 2 * maximum(Ns)
data = load_ett(data_path; n_rows=max_rows)
@info "Loaded ETT data" size=size(data) path=pc["data_path"]

output_dir = joinpath(@__DIR__, "..", "data", "chain_start_sensitivity")
mkpath(output_dir)

@info "Warming up (compilation)…"
let s = build_setup_start(data, 50, 50, p_drop, first(seeds), 1, D, L, input_cols, output_cols)
    RxBayesOpt.forecast_ss_raw(s, D, L, ℓs, γ2s, R_diag_init)
    RxBayesOpt.forecast_km(s, D, L, ℓs, γ2s, R_diag_init)
end

records = Dict{String,Any}[]        # per (N, start, seed)
chainq  = Dict{Tuple{Int,Int},Any}()  # (N,start) -> chain quantities (seed-independent)
orders_by_N = Dict{Int,Vector{Vector{Int}}}()

for N in Ns
    starts = unique(round.(Int, range(1, 2N, length=n_starts)))
    orders_by_N[N] = Vector{Int}[]
    run_km = N <= km_max_N
    @info "ETT chain-start sweep" C=2N n_starts=length(starts) run_km=run_km
    for start in starts
        chain_done = false
        for seed in seeds
            setup = build_setup_start(data, N, N, p_drop, seed, start,
                                      D, L, input_cols, output_cols)
            if !chain_done
                cd = chain_distortion(setup.Xo)
                cq = nn_chain_quality(setup.Xo)
                chainq[(N, start)] = (; T_pi=cd.T_pi, eps_pi=cd.eps_pi,
                                       mean_delta=cq.mean_delta, max_delta=cq.max_delta,
                                       dist_norm=setup.dist_norm)
                push!(orders_by_N[N], setup.order)
                chain_done = true
            end
            ss = RxBayesOpt.forecast_ss_raw(setup, D, L, ℓs, γ2s, R_diag_init)
            km = run_km ? RxBayesOpt.forecast_km(setup, D, L, ℓs, γ2s, R_diag_init) :
                          (mnll=NaN, rmse=NaN, time=NaN)
            push!(records, Dict(
                "N"=>N, "C"=>2N, "start"=>start, "seed"=>seed,
                "ss_rmse"=>ss.rmse, "ss_mnll"=>ss.mnll, "ss_time"=>ss.time,
                "km_rmse"=>km.rmse, "km_mnll"=>km.mnll, "km_time"=>km.time))
        end
    end
end

# ─── Per-(N,start) medians over seeds ────────────────────────────────────────
perstart = Dict{Int,Vector{Dict{String,Any}}}()
for N in Ns
    starts = unique(round.(Int, range(1, 2N, length=n_starts)))
    rows = Dict{String,Any}[]
    for start in starts
        sub = [r for r in records if r["N"] == N && r["start"] == start]
        cq = chainq[(N, start)]
        push!(rows, Dict(
            "start"=>start,
            "ss_rmse"=>_med(Float64[r["ss_rmse"] for r in sub]),
            "ss_mnll"=>_med(Float64[r["ss_mnll"] for r in sub]),
            "km_rmse"=>_med(Float64[r["km_rmse"] for r in sub]),
            "km_mnll"=>_med(Float64[r["km_mnll"] for r in sub]),
            "T_pi"=>cq.T_pi, "eps_pi"=>cq.eps_pi,
            "mean_delta"=>cq.mean_delta, "max_delta"=>cq.max_delta))
    end
    perstart[N] = rows
end

# ─── Save raw records ────────────────────────────────────────────────────────
_safe(x) = (x isa AbstractFloat && !isfinite(x)) ? nothing : x
_safe(d::Dict) = Dict(k => _safe(v) for (k, v) in d)
_safe(v::AbstractVector) = [_safe(x) for x in v]

open(joinpath(output_dir, "comparison.json"), "w") do io
    JSON.print(io, Dict("p"=>p_drop, "seeds"=>seeds, "n_starts"=>n_starts,
                        "Ns"=>Ns, "km_max_N"=>km_max_N,
                        "records"=>_safe(records)), 2)
end
@info "Saved raw per-(start,seed) records to comparison.json"

# ─── Per-C summary ───────────────────────────────────────────────────────────
summary = Dict{String,Any}()
for N in Ns
    rows = perstart[N]
    col(k) = Float64[r[k] for r in rows]
    T = col("T_pi"); eps = col("eps_pi")
    ssr = col("ss_rmse"); ssm = col("ss_mnll")
    summary["C=$(2N)"] = Dict(
        "N"=>N, "n_starts"=>length(rows),
        "ss_rmse"=>_stats(ssr), "ss_mnll"=>_stats(ssm),
        "km_rmse"=>_stats(col("km_rmse")), "km_mnll"=>_stats(col("km_mnll")),
        "T_pi"=>_stats(T), "eps_pi"=>_stats(eps),
        "edge_jaccard"=>_edge_jaccard(orders_by_N[N]),
        "corr_ssrmse_eps"=>Dict(pairs(_boot_cor_ci(eps, ssr))...),
        "corr_ssmnll_eps"=>Dict(pairs(_boot_cor_ci(eps, ssm))...),
        "corr_ssrmse_T"  =>Dict(pairs(_boot_cor_ci(T, ssr))...),
    )
end

# ─── Start-scheme robustness check ───────────────────────────────────────────
# Does the small spread survive an adversarial choice of starts? Compare even,
# random, and farthest-point (maximally diverse) start schemes. If diverse starts
# neither lower the edge-Jaccard nor raise the spread, the greedy chain is
# self-correcting w.r.t. its start and the reported spread is representative, not
# a lucky lower bound. Uses a few seeds (SS-LMC only — KM is order-invariant).
scheme_seeds = seeds[1:min(3, length(seeds))]
scheme_summary = Dict{String,Any}()
for N in Ns
    C = 2N
    X = _ett_inputs(data, N, input_cols)
    schemes = ("even" => _even_starts(C, n_starts),
               "random" => _random_starts(C, n_starts, 123),
               "diverse" => _farthest_starts(X, n_starts))
    @info "ETT start-scheme check" C=C n_scheme_seeds=length(scheme_seeds)
    for (name, starts) in schemes
        pr = Float64[]; pm = Float64[]; ords = Vector{Int}[]
        for start in starts
            rs = Float64[]; ms = Float64[]
            for (k, sd) in enumerate(scheme_seeds)
                setup = build_setup_start(data, N, N, p_drop, sd, start,
                                          D, L, input_cols, output_cols)
                k == 1 && push!(ords, setup.order)
                ss = RxBayesOpt.forecast_ss_raw(setup, D, L, ℓs, γ2s, R_diag_init)
                push!(rs, ss.rmse); push!(ms, ss.mnll)
            end
            push!(pr, _med(rs)); push!(pm, _med(ms))
        end
        prf = _finite(pr); pmf = _finite(pm)
        scheme_summary["C=$C/$name"] = Dict(
            "edge_jaccard" => _edge_jaccard(ords),
            "rmse_std" => length(prf) > 1 ? std(prf) : 0.0,
            "mnll_std" => length(pmf) > 1 ? std(pmf) : 0.0,
            "rmse_mean" => isempty(prf) ? nothing : mean(prf))
    end
end

metrics = Dict("p"=>p_drop, "n_seeds"=>length(seeds), "n_starts"=>n_starts,
               "Ns"=>Ns, "Cs"=>[2N for N in Ns], "per_C"=>summary,
               "scheme_seeds"=>scheme_seeds, "scheme_comparison"=>scheme_summary)
open(joinpath(output_dir, "metrics.json"), "w") do io
    JSON.print(io, _safe(metrics), 2)
end
@info "Saved per-C variance summary to metrics.json"

# ─── Plots ───────────────────────────────────────────────────────────────────
try
    _spread_plot(perstart, Ns, "ss_rmse", "km_rmse", "Forecast RMSE",
                 "rmse_spread_vs_C", output_dir)
    _spread_plot(perstart, Ns, "ss_mnll", "km_mnll", "Forecast MNLL",
                 "mnll_spread_vs_C", output_dir)
    _std_vs_C_plot(summary, Ns, output_dir)
    _distortion_plot(perstart, Ns, output_dir)
    _scheme_plot(scheme_summary, Ns, output_dir)
catch e
    @warn "Plotting failed (results are saved in JSON)" exception=e
end

@info "ETT chain starting-point sensitivity (Experiment A) complete"
for N in Ns
    s = summary["C=$(2N)"]
    @info "C=$(2N)" ss_rmse_mean=s["ss_rmse"]["mean"] ss_rmse_std=s["ss_rmse"]["std"] ss_rmse_cv=s["ss_rmse"]["cv"] km_rmse_std=s["km_rmse"]["std"] ss_mnll_std=s["ss_mnll"]["std"] edge_jaccard=s["edge_jaccard"]
end
