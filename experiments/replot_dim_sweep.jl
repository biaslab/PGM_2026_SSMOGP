# Re-render the dim_sweep figures from cached comparison.json with the ETT
# export style (size=(560,380), default fonts, lw=2), so Figure 3 matches the
# other \resizebox'd figures in the paper. No experiment re-run required.
#
# Run: julia --project=. experiments/replot_dim_sweep.jl <comparison.json> <output_dir>
using JSON, Statistics
import Plots
using Plots: plot, plot!, savefig, gr, mm

const PGF_OK = try
    @eval using PGFPlotsX
    true
catch
    false
end

json_path  = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "figures", "dim_sweep", "comparison.json")
output_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "..", "figures", "dim_sweep")

results = JSON.parsefile(json_path)               # Vector of Dict{String,Any}
ds      = sort(unique(Int(r["d"]) for r in results))
@info "Loaded $(length(results)) records; ds=$ds; out=$output_dir"

_nanmean(x) = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
_nanstd(x)  = (v = filter(!isnan, x); length(v) > 1 ? std(v) : 0.0)

plot_kw = (; size=(560, 380), left_margin=8mm, bottom_margin=6mm)

# Save both PNG (gr) and TikZ (pgfplotsx) from the same builder, mirroring save_plot.
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

methods = ["ss", "km", "svgp"]
labels  = ["SS-LMC", "KM-LMC", "SVGP-LMC"]
colors  = [:blue, :red, :green]

function metric_vs_d(metric, ylabel, fname; legendpos=:topleft, yscale=:identity)
    save_both(fname) do
        p = plot(; xlabel="Input dimension M", ylabel=ylabel,
                 legend=legendpos, xscale=:log2, yscale=yscale, plot_kw...)
        for (m, lab, col) in zip(methods, labels, colors)
            means = Float64[]; stds = Float64[]
            for d in ds
                runs = filter(r -> Int(r["d"]) == d, results)
                vals = Float64[r[m][metric] for r in runs if haskey(r, m)]
                push!(means, _nanmean(vals)); push!(stds, _nanstd(vals))
            end
            ribbon_vals = yscale == :log10 ? min.(stds, means .* 0.99) : stds
            plot!(p, ds, means, ribbon=ribbon_vals, fillalpha=0.15, lw=2,
                  marker=:circle, label=lab, color=col)
        end
        p
    end
end

metric_vs_d("rmse", "Held-out RMSE", "rmse_vs_d")
metric_vs_d("mnll", "Held-out MNLL", "mnll_vs_d")

# Chain quality vs d (mean Δ and max Δ).
save_both("chain_quality_vs_d") do
    p = plot(; xlabel="Input dimension M", ylabel="Chain Δ (consecutive distance)",
             legend=:topleft, xscale=:log2, plot_kw...)
    mean_curve = Float64[]; mean_std = Float64[]
    max_curve  = Float64[]; max_std  = Float64[]
    for d in ds
        runs  = filter(r -> Int(r["d"]) == d, results)
        mvals = Float64[r["chain"]["mean_delta"] for r in runs]
        xvals = Float64[r["chain"]["max_delta"]  for r in runs]
        push!(mean_curve, _nanmean(mvals)); push!(mean_std, _nanstd(mvals))
        push!(max_curve,  _nanmean(xvals)); push!(max_std,  _nanstd(xvals))
    end
    plot!(p, ds, mean_curve, ribbon=mean_std, fillalpha=0.15, lw=2,
          marker=:circle, label="mean Δ", color=:purple)
    plot!(p, ds, max_curve, ribbon=max_std, fillalpha=0.15, lw=2,
          marker=:diamond, label="max Δ", color=:orange, linestyle=:dash)
    p
end

@info "Done."
