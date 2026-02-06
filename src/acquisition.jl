"""
    ucb_acquisition(res, state, cfg, N) -> (ucb, μ_pred, σ_pred)

Compute UCB acquisition values from RxInfer inference results.

Returns:
- `ucb`: scalarized UCB values for all N points
- `μ_pred`: predicted means (original scale), length-N vector of vectors
- `σ_pred`: predicted stds (original scale), length-N vector of vectors
"""
function ucb_acquisition(res, state::BOState, cfg::ExperimentConfig, N::Int)
    pred_output = last(res.posteriors[:my])
    μ_pred_std = mean.(pred_output)
    σ_pred_std = [sqrt.(var(pred_output[i])) for i in 1:N]

    μ_pred = [μ_pred_std[i] .* state.σy .+ state.μy for i in 1:N]
    σ_pred = [σ_pred_std[i] .* state.σy for i in 1:N]

    D = cfg.D
    s = cfg.s
    μs = [dot(s, μ_pred[i]) for i in 1:N]
    σs = [sqrt(sum((s[j] * σ_pred[i][j])^2 for j in 1:D)) for i in 1:N]

    ucb = μs .+ cfg.β .* σs
    (; ucb, μ_pred, σ_pred, μs, σs)
end

"""
    select_next_point(ucb, Y) -> Int

Select the next point to evaluate: the unobserved point with highest UCB.
Returns `0` if all points are observed.
"""
function select_next_point(ucb::AbstractVector, Y::AbstractVector)
    observed = findall(!ismissing, Y)
    unobs = setdiff(1:length(Y), observed)
    isempty(unobs) && return 0
    unobs[argmax(ucb[unobs])]
end
