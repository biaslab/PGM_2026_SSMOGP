# Standard benchmark functions for multi-output Bayesian optimization experiments.
#
# Functions:
# - hartmann6:          Standard Hartmann 6-dimensional function
# - make_mo_hartmann:   Multi-output Hartmann benchmark (d=6, configurable D)
# - make_environmental: Environmental monitoring benchmark (d=4, D = n_spatial * n_temporal)

"""
    _sigmoid(x) -> Float64

Logistic sigmoid, maps R -> (0, 1). Used to transform standardized inputs to [0,1].
"""
@inline _sigmoid(x::Real) = 1.0 / (1.0 + exp(-x))

"""
    hartmann6(x::AbstractVector{<:Real}) -> Float64

Hartmann 6-dimensional function on [0, 1]^6.

Global maximum ≈ 3.3224 at x* ≈ (0.201, 0.150, 0.477, 0.275, 0.312, 0.657).

Reference: Dixon & Szego (1978), "The Global Optimization Problem".
"""
function hartmann6(x::AbstractVector{<:Real})
    α = [1.0, 1.2, 3.0, 3.2]
    A = [10.0  3.0  17.0  3.5  1.7  8.0;
          0.05 10.0  17.0  0.1  8.0 14.0;
          3.0  3.5   1.7 10.0 17.0  8.0;
         17.0  8.0   0.05 10.0  0.1 14.0]
    P = 1e-4 * [1312.0 1696.0 5569.0  124.0 8283.0 5886.0;
                2329.0 4135.0 8307.0 3736.0 1004.0 9991.0;
                2348.0 1451.0 3522.0 2883.0 3047.0 6650.0;
                4047.0 8828.0 8732.0 5743.0 1091.0  381.0]

    val = 0.0
    for i in 1:4
        inner = 0.0
        for j in 1:6
            inner += A[i, j] * (x[j] - P[i, j])^2
        end
        val += α[i] * exp(-inner)
    end
    val
end

# Fixed input permutations for multi-output Hartmann latent functions
const _HARTMANN_PERMS = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 1],
    [6, 5, 4, 3, 2, 1],
    [3, 1, 5, 2, 6, 4],
    [4, 6, 2, 5, 1, 3],
    [5, 4, 1, 6, 3, 2],
]

"""
    make_mo_hartmann(; D, Q=min(D,6), W_seed=42) -> Function

Create a multi-output Hartmann-6 benchmark (d=6, D outputs).

Q latent functions are Hartmann-6 evaluated on permuted inputs. A fixed
mixing matrix W (D x Q) produces D correlated outputs. This creates a
function with genuine LMC-like correlation structure.

Inputs should be standardized (approximately N(0,1)); internally mapped
to [0,1]^6 via sigmoid.

Reference: Based on Hartmann-6 (Dixon & Szego, 1978), extended to multi-output
via LMC-style mixing as in Alvarez et al. (2012).
"""
function make_mo_hartmann(; D::Int, Q::Int=min(D, 6), W_seed::Int=42)
    @assert 1 <= Q <= 6 "Q must be between 1 and 6 (available permutations)"

    # Fixed mixing matrix (reproducible across runs)
    rng = MersenneTwister(W_seed)
    W_true = randn(rng, D, Q) * 0.3
    # Ensure each output has at least one strong loading
    for d in 1:D
        q = ((d - 1) % Q) + 1
        W_true[d, q] += 0.7
    end

    function(x::AbstractVector{<:Real})
        x_unit = [_sigmoid(x[j]) for j in 1:6]
        f = [hartmann6(x_unit[_HARTMANN_PERMS[q]]) for q in 1:Q]
        W_true * f
    end
end

"""
    make_environmental(; n_spatial=3, n_temporal=4) -> Function

Environmental monitoring benchmark (d=4, D = n_spatial x n_temporal).

Models pollutant concentration from a point source in a 1D medium with
diffusion and exponential decay. Outputs at different spatial-temporal
locations are naturally correlated through the shared physics.

Inputs (d=4, standardized, mapped internally to physical ranges):
- x[1] -> M  in [7, 13]:          source mass
- x[2] -> Dc in [0.02, 0.12]:     diffusivity
- x[3] -> L  in [0.01, 3.0]:      source location
- x[4] -> tau in [30.01, 30.295]:  decay time constant

Reference:
- Bliznyuk et al. (2008), Bayesian calibration of mechanistic aquatic biogeochemical models
- Maddox et al. (2021), Bayesian Optimization with High-Dimensional Outputs, NeurIPS
"""
function make_environmental(; n_spatial::Int=3, n_temporal::Int=4)
    s_locs = collect(range(0.0, 3.0, length=n_spatial))
    t_locs = collect(range(15.0, 60.0, length=n_temporal))
    D_out = n_spatial * n_temporal

    function(x::AbstractVector{<:Real})
        u = [_sigmoid(x[j]) for j in 1:4]
        M  = 7.0   + 6.0   * u[1]
        Dc = 0.02  + 0.10  * u[2]
        L  = 0.01  + 2.99  * u[3]
        τ  = 30.01 + 0.285 * u[4]

        y = zeros(D_out)
        idx = 0
        for t in t_locs
            for s in s_locs
                idx += 1
                denom = sqrt(4π * Dc * t)
                c_direct  = exp(-(s - L)^2 / (4Dc * t)) / denom
                c_reflect = exp(-(s + L)^2 / (4Dc * t)) / denom
                y[idx] = M * (c_direct + c_reflect) * exp(-t / τ)
            end
        end
        y
    end
end

"""
    make_synthetic_1d(; D=6, Q=3, W_seed=42) -> Function

1D multi-output synthetic benchmark based on sinusoidal latent functions.

Q latent functions are sinusoids with distinct frequencies and phases:
`f_q(u) = sin(q * u + (q-1) * π/Q)` for q = 1..Q, where u ∈ [0, 2π].

A fixed mixing matrix W (D × Q) produces D correlated outputs. In 1D the
NN chain ordering is the natural ordering, so the SS-GP is exact — useful
for cleanly isolating scaling comparisons without chain ordering artifacts.

Input: single standardized scalar, mapped internally to [0, 2π] via sigmoid.
Output: D-dimensional vector.
"""
function make_synthetic_1d(; D::Int=6, Q::Int=3, W_seed::Int=42)
    rng = MersenneTwister(W_seed)
    W_true = randn(rng, D, Q) * 0.3
    for d in 1:D
        q = ((d - 1) % Q) + 1
        W_true[d, q] += 0.7
    end

    function(x::AbstractVector{<:Real})
        u = _sigmoid(x[1]) * 2π
        f = [sin(q * u + (q - 1) * π / Q) for q in 1:Q]
        W_true * f
    end
end
