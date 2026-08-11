# Paper figures for the chain starting-point sensitivity study (two panels meant
# to sit side by side):
#
#   (a) spread_vs_M — sensitivity vs input dimension M   (Experiment B, synthetic)
#   (b) spread_vs_C — sensitivity vs chain length C      (Experiment A, ETT)
#
# Both plot the same quantity on a log y-axis: the standard deviation ACROSS
# STARTS of the per-start (median-over-seeds) held-out RMSE, for SS-LMC.
#
# Error bars are a percentile bootstrap 95% CI for that std, resampling the
# per-start values — the y-value is itself an estimated dispersion, so its own
# sampling error is the honest uncertainty to show. The bars are wide by
# construction: a std from n starts carries a relative SE of roughly
# 1/sqrt(2(n-1)) (~24% at n=10, ~16% at n=20), so only large ratios between
# x-values should be read as real.
#
# The y-limits are NOT shared between the two figures: both are RMSE spreads but
# in different units (synthetic sensor-network outputs vs raw ETT loads), so
# their absolute levels are not commensurable.
#
# Replots from cached JSON — no experiment re-run needed.
# Run: julia --project=. experiments/plot_chain_start_paper.jl [dim_dir] [sens_dir] [output_dir]
# Or:  dvc repro chain_start_paper

using JSON, Statistics, Random, TuePlots
import Plots
using Plots: plot, plot!, savefig, gr, mm

const PGF_OK = try
    @eval using PGFPlotsX
    true
catch
    false
end

const ROOT = joinpath(@__DIR__, "..")

dim_dir    = length(ARGS) >= 1 ? ARGS[1] : joinpath(ROOT, "data", "chain_start_dim")
sens_dir   = length(ARGS) >= 2 ? ARGS[2] : joinpath(ROOT, "data", "chain_start_sensitivity")
output_dir = length(ARGS) >= 3 ? ARGS[3] : joinpath(ROOT, "data", "chain_start_paper")
mkpath(output_dir)

# ─── Save both PNG (gr) and TikZ (pgfplotsx) from one builder ────────────────
# Mirrors RxBayesOpt.save_plot without loading the module (which pulls RxInfer).
function save_both(build, stem)
    gr()
    savefig(build(), joinpath(output_dir, stem) * ".png")
    @info "Saved $stem.png"
    PGF_OK || (@warn "PGFPlotsX unavailable; skipped $stem.tikz"; return)
    try
        Plots.pgfplotsx()
        savefig(build(), joinpath(output_dir, stem) * ".tikz")
        @info "Saved $stem.tikz"
    catch e
        @warn "tikz failed for $stem" exception=e
    finally
        gr()
    end
end

# Warm up the PGFPlotsX backend: the first render after switching backends can
# emit an empty tikz document, so burn that on a throwaway plot.
if PGF_OK
    try
        Plots.pgfplotsx(); savefig(plot(1:2, 1:2), tempname() * ".tikz")
    catch
    finally
        gr()
    end
end

# ─── Aggregation (mirrors _med/_stats in the experiment scripts) ──────────────
# JSON stores non-finite metrics as null, so drop `nothing` alongside NaN/Inf.
_num(v) = v === nothing ? NaN : Float64(v)
_finite(v) = filter(isfinite, v)
_med(v) = (f = _finite(v); isempty(f) ? NaN : median(f))
_std(v) = (f = _finite(v); length(f) > 1 ? std(f) : (isempty(f) ? NaN : 0.0))

# The per-start values: median over seeds for each start. Their std across
# starts is the published statistic; keeping the vector lets us bootstrap it.
function per_start_values(records, group_key, group_val, metric)
    sub = [r for r in records if Int(r[group_key]) == group_val]
    starts = sort(unique(Int(r["start"]) for r in sub))
    seeds = length(unique(Int(r["seed"]) for r in sub))
    vals = Float64[_med(Float64[_num(r[metric]) for r in sub if Int(r["start"]) == s])
                   for s in starts]
    (; vals, n_starts=length(starts), n_seeds=seeds)
end

# Percentile bootstrap 95% CI for the std across starts, resampling starts.
# Seeded so the figure is reproducible.
function boot_std_ci(vals::Vector{Float64}; nboot::Int=4000, seed::Int=0)
    f = _finite(vals)
    length(f) > 1 || return (NaN, NaN)
    rng = MersenneTwister(seed)
    n = length(f)
    stds = Float64[std(f[rand(rng, 1:n, n)]) for _ in 1:nboot]
    (quantile(stds, 0.025), quantile(stds, 0.975))
end

# metrics.json holds the numbers quoted in chain_start_results.md; comparison.json
# holds the raw per-(start,seed) records. Read the former, verify against the
# latter, so a figure can never silently drift from the write-up.
function checked_spread(dir, summary_key, group_key, group_label, xs, metric)
    m = JSON.parsefile(joinpath(dir, "metrics.json"))
    records = JSON.parsefile(joinpath(dir, "comparison.json"))["records"]
    stds = Float64[]; los = Float64[]; his = Float64[]
    n_starts = Int[]; n_seeds = Int[]
    for x in xs
        ps = per_start_values(records, group_key, x, metric)
        published = _num(m[summary_key]["$group_label$x"][metric]["std"])
        recomputed = _std(ps.vals)
        if !(isnan(published) && isnan(recomputed)) && !isapprox(published, recomputed; atol=1e-10, rtol=1e-8)
            @warn "metrics.json disagrees with comparison.json" dir metric x published recomputed
        end
        lo, hi = boot_std_ci(ps.vals)
        push!(stds, published); push!(los, lo); push!(his, hi)
        push!(n_starts, ps.n_starts); push!(n_seeds, ps.n_seeds)
    end
    length(unique(n_starts)) == 1 && length(unique(n_seeds)) == 1 ||
        @warn "start/seed counts vary across x" dir n_starts n_seeds
    (; stds, los, his, n_starts=first(n_starts), n_seeds=first(n_seeds))
end

# ─── Shared style ────────────────────────────────────────────────────────────
# One series per figure: SS-LMC held-out RMSE spread. Limits are per-figure, not
# shared — the two RMSE spreads are in different units (see the header).
const YLIM_M = (5e-5, 5e-3)
const YLIM_C = (1e-3, 1e-2)

# Paper typography: the same TuePlots theme `RxBayesOpt.publication_theme_kwargs`
# applies (ProbNum25 settings — ~241x149 pt canvas, 9 pt labels, 7 pt ticks),
# so these figures carry body-text-sized fonts like the paper's other figures
# once \input at close to their natural width.
const THEME = TuePlots.get_plotsjl_theme_kwargs(
    TuePlots.SETTINGS[:ProbNum25]; font=false, fontsize=true, figsize=true,
    single_column=true, nrows=1, ncols=1, thickness_scaling=true)

const PLOT_KW = (; THEME..., yscale=:log10, left_margin=2mm, bottom_margin=1mm,
                 lw=2, legend=false, color=:blue, marker=:circle, markersize=3)

# Bootstrap CI as an asymmetric error bar: Plots wants distances from the point.
_yerr(s, lo, hi) = (max.(s .- lo, 0.0), max.(hi .- s, 0.0))

# Multiplicative x-padding (the axes are log-scaled) so the first/last error
# bars are not drawn flush against the frame.
_xpad(xs, f) = (first(xs) / f, last(xs) * f)

# Sample size stated on the figure itself, since the y-value is a spread and its
# credibility depends entirely on how many starts it was computed from.
_nlabel(r) = "$(r.n_starts) starts × $(r.n_seeds) seeds"

# Compact one-line log of the plotted numbers (Julia truncates raw vectors).
_fmt(r) = join(("$(round(s, sigdigits=3)) [$(round(l, sigdigits=3)), $(round(h, sigdigits=3))]"
                for (s, l, h) in zip(r.stds, r.los, r.his)), "  ")

# ─── (a) Sensitivity vs input dimension M (Experiment B, synthetic) ──────────
ds = Int.(JSON.parsefile(joinpath(dim_dir, "comparison.json"))["ds"])
dim_r = checked_spread(dim_dir, "per_M", "M", "M=", ds, "ss_rmse")
@info "Experiment B RMSE spread (std [95% CI])" ds n=_nlabel(dim_r) vals=_fmt(dim_r)

save_both("spread_vs_M") do
    xs = Float64.(ds)
    p = plot(xs, dim_r.stds;
             yerror=_yerr(dim_r.stds, dim_r.los, dim_r.his),
             xlabel="Input dimension M", ylabel="RMSE spread [output units]",
             xscale=:log2, xticks=(xs, string.(ds)), xlim=_xpad(xs, 1.35),
             ylim=YLIM_M, PLOT_KW...)
    Plots.annotate!(p, xs[1], YLIM_M[2] / 1.5,
                    Plots.text(_nlabel(dim_r), 7, :left, :gray))
    p
end

# ─── (b) Sensitivity vs chain length C (Experiment A, ETT) ───────────────────
Cs = Int.(JSON.parsefile(joinpath(sens_dir, "comparison.json"))["Ns"]) .* 2
sens_r = checked_spread(sens_dir, "per_C", "C", "C=", Cs, "ss_rmse")
@info "Experiment A RMSE spread (std [95% CI])" Cs n=_nlabel(sens_r) vals=_fmt(sens_r)

save_both("spread_vs_C") do
    xs = Float64.(Cs)
    p = plot(xs, sens_r.stds;
             yerror=_yerr(sens_r.stds, sens_r.los, sens_r.his),
             xlabel="Candidate chain length C", ylabel="RMSE spread [ETT load units]",
             xscale=:log10, xticks=(xs, string.(Cs)), xlim=_xpad(xs, 1.15),
             ylim=YLIM_C, PLOT_KW...)
    Plots.annotate!(p, xs[1], YLIM_C[2] / 1.2,
                    Plots.text(_nlabel(sens_r), 7, :left, :gray))
    p
end

@info "Done." output_dir
