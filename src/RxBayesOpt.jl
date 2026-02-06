module RxBayesOpt

using RxInfer, Distributions, LinearAlgebra, Random, Statistics
using Plots, Optim

include("types.jl")
include("utils.jl")
include("statespace.jl")
include("model.jl")
include("acquisition.jl")
include("hyperparameters.jl")
include("visualization.jl")
include("bo.jl")

export ExperimentConfig, BOState, BOResult
export print_config, print_summary
export row, sqdist, nn_chain_order, blockdiag
export matern32_blocks_from_Δ, additive_multioutput_blocks_from_Δ
export additive_gp_vv
export ucb_acquisition, select_next_point
export log_marginal_likelihood, tune_hyperparameters
export plot_bo_step
export setup_experiment, run_bo!

end # module RxBayesOpt
