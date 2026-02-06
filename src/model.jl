# RxInfer probabilistic model for the additive multi-output state-space GP.
#
# Defines a linear dynamical system:
# - f[0] ~ MvNormal(0, P)       (stationary prior)
# - f[i] ~ MvNormal(A[i]·f[i-1], Q[i])  (state transition)
# - Y[i] ~ MvNormal(H·f[i], R)  (observation)
#
# Missing entries in Y are predicted by RxInfer's message passing.

@model function additive_gp_vv(Y, P, A, Q, H, R)
    fprev ~ MvNormal(μ=zeros(size(P, 1)), Σ=P)
    for i in eachindex(Y)
        f[i] ~ MvNormal(μ=A[i] * fprev, Σ=Q[i])
        my[i] := H * f[i]
        Y[i] ~ MvNormal(μ=my[i], Σ=R)
        fprev = f[i]
    end
end
