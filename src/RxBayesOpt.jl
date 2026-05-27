module RxBayesOpt

using RxInfer, Distributions, LinearAlgebra, Random, Statistics
using Plots, Optim, JSON
using PGFPlotsX
using TuePlots

include("types.jl")
include("utils.jl")
include("statespace.jl")
include("model.jl")
include("acquisition.jl")
include("hyperparameters.jl")
include("visualization.jl")
include("bo.jl")
include("baseline.jl")
include("partial_obs.jl")
include("sequential_design.jl")
include("benchmarks.jl")
include("dim_sweep.jl")
include("ett.jl")

export AbstractBOState, ExperimentConfig, BOResult
export POState, BaselinePOState, BaselineState
export print_config, print_summary
export row, sqdist, nn_chain_order, nn_chain_quality, blockdiag
export matern32_blocks_from_Δ, additive_multioutput_blocks_from_Δ
export additive_gp_po
export ucb_acquisition, select_next_point
export log_marginal_likelihood, tune_hyperparameters
export save_plot, plot_bo_step, publication_theme_kwargs
export setup_experiment, save_results
export setup_baseline, run_bo_baseline!
export setup_po, run_bo_po!
export setup_baseline_po, run_bo_baseline_po!
export load_ett, run_ett_forecast, run_ett_sweeps
export hartmann6, make_mo_hartmann, make_environmental, make_synthetic_1d, make_sensor_network
export run_dim_sweep
export SDResult
export variance_acquisition, baseline_po_variance_acquisition
export run_sd_po!, run_sd_baseline_po!, run_sd_comparison
export run_bq_comparison

end # module RxBayesOpt
