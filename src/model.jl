# RxInfer probabilistic model for the additive multi-output state-space GP
# with online observation noise learning.
#
# Defines a linear dynamical system with a learnable noise covariance:
# - R ~ prior_R                  (InverseWishart prior on observation noise)
# - f[0] ~ MvNormal(0, P)       (stationary prior)
# - f[i] ~ MvNormal(A[i]·f[i-1], Q[i])  (state transition)
# - Y[i] ~ MvNormal(H·f[i], R)  (observation)
#
# The posterior over R is inferred via variational message passing and
# fed back as the prior for the next BO step, enabling online noise adaptation.
# Missing entries in Y are predicted by RxInfer's message passing.

@model function additive_gp_vv(Y, P, A, Q, H, prior_R)
    R ~ prior_R
    fprev ~ MvNormal(μ=zeros(size(P, 1)), Σ=P)
    for i in eachindex(Y)
        f[i] ~ MvNormal(μ=A[i] * fprev, Σ=Q[i])
        my[i] := H * f[i]
        Y[i] ~ MvNormal(μ=my[i], Σ=R)
        fprev = f[i]
    end
end

RxInfer.GraphPPL.default_constraints(::typeof(additive_gp_vv)) = @constraints begin
    q(R, my) = q(R)q(my)
end