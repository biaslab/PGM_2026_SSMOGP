# RxInfer probabilistic model for the additive multi-output state-space GP
# with per-output scalar observation noise.
#
# Defines a linear dynamical system:
# - f[0] ~ MvNormal(0, P)               (stationary prior)
# - f[i] ~ MvNormal(A[i]·f[i-1], Q[i])  (state transition)
# - Y[(i-1)*D + d] ~ NormalMeanPrecision(dot(e_d, H·f[i]), τ[d])  (per-output obs)
#
# Each output d at each chain position i is an independent scalar observation.
# Missing entries in Y propagate naturally via message passing.
# No VMP constraints needed (all noise precisions are fixed).

@model function additive_gp_po(Y, P, A, Q, H, τ, e_vecs, N, D)
    fprev ~ MvNormal(μ=zeros(size(P, 1)), Σ=P)
    for i in 1:N
        f[i] ~ MvNormal(μ=A[i] * fprev, Σ=Q[i])
        my[i] := H * f[i]
        for d in 1:D
            Y[(i-1)*D + d] ~ NormalMeanPrecision(dot(e_vecs[d], my[i]), τ[d])
        end
        fprev = f[i]
    end
end
