# Correctness tests for the Vecchia/NNGP LMC baseline (`src/vecchia.jl`).
#
# The defining property: Vecchia is an approximation ONLY through the size of
# the conditioning sets. With `m` large enough that every point conditions on
# all observed locations, the conditional factorization is exact, so both the
# predictive marginals and the log-likelihood must agree with the dense KM-LMC
# baseline to machine precision. Everything else (orderings, neighbour search,
# partial-observation handling) is checked around that anchor.
#
# Run with: julia --project=. test/test_vecchia.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Test, Random, Statistics, LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "RxBayesOpt.jl"))
using .RxBayesOpt

const V = RxBayesOpt

"""Small shared fixture: C points in M dims, D outputs, Q latents."""
function _fixture(; N=40, d=2, D=3, Q=2, seed=0, R=0.1)
    cfg = ExperimentConfig(;
        N=N, d=d, Q=Q, D=D,
        ℓs=[1.0, 2.0], σ2s=[2.0, 1.0],
        β=2.0, s=fill(1.0 / D, D),
        n_seed=5, steps=1, R_diag_init=R,
        animate=false, log_every=100, seed=seed,
        obs_pattern=:full, obs_frac=1.0,
    )
    eval_fn = make_sensor_network(; d=d, D=D)
    sd = setup_experiment(cfg, eval_fn)
    (; cfg, sd)
end

"""Mask out a random subset of (point, output) cells; `Y[i]` stays non-missing."""
function _random_mask(N, D, frac_missing, seed)
    rng = MersenneTwister(seed)
    mask = trues(N, D)
    for i in 1:N, d in 1:D
        rand(rng) < frac_missing && (mask[i, d] = false)
    end
    mask
end

@testset "Vecchia / NNGP LMC baseline" begin

    @testset "maxmin ordering" begin
        rng = MersenneTwister(3)
        X = rand(rng, 60, 2)
        idx = collect(1:60)
        ord = V._maxmin_order(X, idx)

        @test length(ord) == 60
        @test sort(ord) == idx                      # a permutation

        # Coarse-to-fine: the minimum distance from ord[k] back to its
        # predecessors must be non-increasing in k.
        mins = Float64[]
        for k in 2:length(ord)
            push!(mins, minimum(V.sqdist(V.row(X, ord[k]), V.row(X, ord[j]))
                                for j in 1:k-1))
        end
        @test all(diff(mins) .<= 1e-12)

        # Subset ordering returns indices into the ORIGINAL rows.
        sub = [2, 7, 11, 30, 44]
        @test sort(V._maxmin_order(X, sub)) == sort(sub)

        @test V._maxmin_order(X, Int[]) == Int[]
        @test V._maxmin_order(X, [5]) == [5]
    end

    @testset "m-nearest neighbour search" begin
        X = Float64[0 0; 1 0; 2 0; 3 0; 10 0]
        pool = collect(1:5)

        @test V._m_nearest(X, 1, pool, 3) == [1, 2, 3]           # includes self
        @test V._m_nearest(X, 1, pool, 3; exclude_self=true) == [2, 3, 4]
        @test sort(V._m_nearest(X, 1, pool, 99)) == pool         # m > |pool| clamps
        @test V._m_nearest(X, 1, Int[], 3) == Int[]
        @test V._m_nearest(X, 1, [1], 3; exclude_self=true) == Int[]
    end

    @testset "LMC pair covariance matches the dense kernel" begin
        fx = _fixture(; N=25, d=2, D=3)
        cfg, sd = fx.cfg, fx.sd
        N, D = cfg.N, cfg.D

        # _lmc_full_kernel uses point-major flat indexing (i-1)*D + d.
        Kfull = V._lmc_full_kernel(sd.Xo, sd.W, cfg.ℓs, cfg.σ2s, sd.dist_norm)

        pts  = repeat(1:N, inner=D)
        dims = repeat(1:D, outer=N)
        Kpair = V._lmc_cov_pairs(sd.Xo, pts, dims, pts, dims,
                                 sd.W, cfg.ℓs, cfg.σ2s, sd.dist_norm)

        @test size(Kpair) == size(Kfull)
        @test maximum(abs, Kpair - Kfull) < 1e-10

        # Asymmetric slice: query one point vs. a handful of pairs.
        a_pts, a_dims = fill(4, D), collect(1:D)
        b_pts, b_dims = [1, 1, 7, 9], [1, 2, 3, 1]
        Kab = V._lmc_cov_pairs(sd.Xo, a_pts, a_dims, b_pts, b_dims,
                               sd.W, cfg.ℓs, cfg.σ2s, sd.dist_norm)
        for (ai, (p, dd)) in enumerate(zip(a_pts, a_dims)),
            (bi, (q, ee)) in enumerate(zip(b_pts, b_dims))
            @test Kab[ai, bi] ≈ Kfull[(p-1)*D + dd, (q-1)*D + ee] atol=1e-10
        end
    end

    @testset "prior fallback with no observations" begin
        fx = _fixture(; N=20, d=2, D=3)
        cfg, sd = fx.cfg, fx.sd
        sd_empty = merge(sd, (; Y = Vector{Union{Missing, Vector{Float64}}}(
            fill(missing, cfg.N)),))

        st = setup_vecchia(cfg, sd_empty; m=10)
        pred = V._lmc_vecchia_predict(st)
        prior_sd = sqrt.(V._lmc_self_variance(sd.W, cfg.σ2s))

        @test all(all(iszero, pred.μ_pred[i]) for i in 1:cfg.N)
        @test all(pred.σ_pred[i] ≈ prior_sd for i in 1:cfg.N)
        @test vecchia_loglik(st) == 0.0
    end

    """
    Jitter-free dense GP posterior over all N points given `Y`/`mask`.

    The shipped KM-LMC baseline (`_lmc_predict_po`) regularizes with `1e-8I`,
    which puts a ~1e-8 floor under any comparison against it. This reference
    adds no jitter at all, so it pins Vecchia against the true conditional
    Gaussian rather than against another implementation's rounding.
    """
    function _exact_posterior(cfg, sd, Y, mask)
        N, D = cfg.N, cfg.D
        Kf = V._lmc_full_kernel(sd.Xo, sd.W, cfg.ℓs, cfg.σ2s, sd.dist_norm)

        idx = Int[]; dims = Int[]; yv = Float64[]
        for i in 1:N
            ismissing(Y[i]) && continue
            for d in 1:D
                if mask[i, d]
                    push!(idx, (i-1)*D + d); push!(dims, d); push!(yv, Y[i][d])
                end
            end
        end

        Kobs = Kf[idx, idx] + Diagonal([cfg.R_diag_init for _ in dims])
        L = cholesky(Symmetric(Kobs))
        cross = Kf[:, idx]
        μ_flat = cross * (L \ yv)
        Vc = L.L \ cross'
        var_flat = diag(Kf) .- vec(sum(abs2, Vc; dims=1))

        μ = [μ_flat[(i-1)*D+1 : i*D] for i in 1:N]
        σ = [sqrt.(max.(var_flat[(i-1)*D+1 : i*D], 1e-10)) for i in 1:N]
        (; μ, σ)
    end

    # ── The anchor test ────────────────────────────────────────────────────
    # With m ≥ number of observed locations every conditioning set is the full
    # observed set, so Vecchia collapses to exact GP conditioning and must
    # reproduce the exact posterior to near machine precision.
    @testset "m → n_obs recovers the exact posterior" begin
        for (dd, frac_missing) in [(2, 0.0), (3, 0.0), (2, 0.4)]
            fx = _fixture(; N=30, d=dd, D=3, seed=dd)
            cfg, sd = fx.cfg, fx.sd
            N, D = cfg.N, cfg.D

            # Hold out the last 10 points, mask cells among the observed ones.
            Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
            fill!(Y, missing)
            for k in 1:20
                Y[k] = (sd.Ytrue[k] .- sd.μy) ./ sd.σy
            end
            sd_split = merge(sd, (; Y = Y))
            mask = frac_missing == 0.0 ? trues(N, D) :
                   _random_mask(N, D, frac_missing, 11)

            ref = _exact_posterior(cfg, sd, Y, mask)

            vec_st = setup_vecchia(cfg, sd_split; m=N, mask=mask)
            pv = V._lmc_vecchia_predict(vec_st)

            @test maximum(maximum(abs, pv.μ_pred[i] - ref.μ[i]) for i in 1:N) < 1e-9
            @test maximum(maximum(abs, pv.σ_pred[i] - ref.σ[i]) for i in 1:N) < 1e-9

            # And it agrees with the shipped KM-LMC baseline up to that
            # baseline's own 1e-8 jitter.
            pk = V._lmc_predict_po(setup_baseline_po(cfg, sd_split, mask))
            @test maximum(maximum(abs, pv.μ_pred[i] - pk.μ_pred[i]) for i in 1:N) < 1e-6
            @test maximum(maximum(abs, pv.σ_pred[i] - pk.σ_pred[i]) for i in 1:N) < 1e-6
        end
    end

    @testset "exact-m log-likelihood matches the dense Gaussian log-density" begin
        fx = _fixture(; N=24, d=2, D=3, seed=5)
        cfg, sd = fx.cfg, fx.sd
        N, D = cfg.N, cfg.D

        Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
        for k in 1:N
            Y[k] = (sd.Ytrue[k] .- sd.μy) ./ sd.σy
        end
        sd_obs = merge(sd, (; Y = Y))
        mask = _random_mask(N, D, 0.3, 7)

        st = setup_vecchia(cfg, sd_obs; m=N, mask=mask)
        ll_vecchia = vecchia_loglik(st)

        # Dense reference: N(0, K_obs + R) over the observed entries.
        Kfull = V._lmc_full_kernel(sd.Xo, sd.W, cfg.ℓs, cfg.σ2s, sd.dist_norm)
        idx = Int[]; dims = Int[]; yv = Float64[]
        for i in 1:N, d in 1:D
            if mask[i, d]
                push!(idx, (i-1)*D + d); push!(dims, d); push!(yv, Y[i][d])
            end
        end
        Kobs = Kfull[idx, idx] + Diagonal([cfg.R_diag_init for _ in dims])
        Lref = cholesky(Symmetric(Kobs) + 1e-10I)
        ll_dense = -0.5 * (length(yv) * log(2π) + 2 * sum(log, diag(Lref.L)) +
                           sum(abs2, Lref.L \ yv))

        @test ll_vecchia ≈ ll_dense atol=1e-6
    end

    @testset "accuracy improves monotonically with m" begin
        fx = _fixture(; N=120, d=2, D=3, seed=2)
        cfg, sd = fx.cfg, fx.sd
        N, D = cfg.N, cfg.D

        Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
        fill!(Y, missing)
        train = 1:80
        for k in train
            Y[k] = (sd.Ytrue[k] .- sd.μy) ./ sd.σy
        end
        sd_split = merge(sd, (; Y = Y))
        test_idx = collect(81:N)

        ref = _exact_posterior(cfg, sd, Y, trues(N, D))

        errs = Float64[]
        for m in [1, 3, 10, 30, 80]
            st = setup_vecchia(cfg, sd_split; m=m)
            pv = V._lmc_vecchia_predict(st)
            push!(errs, maximum(maximum(abs, pv.μ_pred[i] - ref.μ[i])
                                for i in test_idx))
        end

        # Error must shrink with m and vanish once m covers the training set.
        @test errs[end] < 1e-9
        @test errs[1] > errs[end]
        @test issorted(errs; rev=true)

        # Vecchia log-likelihood increases toward the exact value as m grows.
        Yall = Vector{Union{Missing, Vector{Float64}}}(undef, N)
        for k in 1:N
            Yall[k] = (sd.Ytrue[k] .- sd.μy) ./ sd.σy
        end
        sd_all = merge(sd, (; Y = Yall))
        lls = [vecchia_loglik(setup_vecchia(cfg, sd_all; m=m)) for m in [1, 5, 20, N]]
        @test issorted(lls)
    end

    @testset "partial observability: masked cells are ignored, not zero-filled" begin
        fx = _fixture(; N=60, d=2, D=3, seed=9)
        cfg, sd = fx.cfg, fx.sd
        N, D = cfg.N, cfg.D

        Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
        for k in 1:N
            Y[k] = (sd.Ytrue[k] .- sd.μy) ./ sd.σy
        end
        sd_obs = merge(sd, (; Y = Y))

        mask = trues(N, D)
        mask[:, 2] .= false                      # output 2 never observed

        st = setup_vecchia(cfg, sd_obs; m=15, mask=mask)
        pred = V._lmc_vecchia_predict(st)

        # Output 2 is still predicted (through LMC coupling) and must not be
        # degenerate or equal to the prior — it borrows strength from 1 and 3.
        prior_sd = sqrt.(V._lmc_self_variance(sd.W, cfg.σ2s))
        @test all(isfinite, reduce(vcat, pred.μ_pred))
        @test any(abs(pred.μ_pred[i][2]) > 1e-6 for i in 1:N)
        @test all(pred.σ_pred[i][2] <= prior_sd[2] + 1e-10 for i in 1:N)

        # Corrupting the masked-out entries must not change any prediction.
        Y2 = [copy(Y[k]) for k in 1:N]
        for k in 1:N
            Y2[k][2] = 1e6
        end
        sd_corrupt = merge(sd, (; Y = Vector{Union{Missing, Vector{Float64}}}(Y2)))
        st2 = setup_vecchia(cfg, sd_corrupt; m=15, mask=mask)
        pred2 = V._lmc_vecchia_predict(st2)

        @test maximum(maximum(abs, pred2.μ_pred[i] - pred.μ_pred[i]) for i in 1:N) < 1e-10
    end

    @testset "acquisition wrappers return original-scale predictions" begin
        fx = _fixture(; N=40, d=2, D=3, seed=4)
        cfg, sd = fx.cfg, fx.sd
        N = cfg.N

        Y = Vector{Union{Missing, Vector{Float64}}}(undef, N)
        fill!(Y, missing)
        for k in 1:25
            Y[k] = (sd.Ytrue[k] .- sd.μy) ./ sd.σy
        end
        sd_split = merge(sd, (; Y = Y))
        st = setup_vecchia(cfg, sd_split; m=12)

        ucb = vecchia_ucb_acquisition(st, cfg)
        @test length(ucb.ucb) == N
        @test all(isfinite, ucb.ucb)

        va = vecchia_variance_acquisition(st, cfg)
        @test length(va.score) == N
        @test all(>=(0), va.score)
        # Same field names as the KM-LMC variance acquisition it stands in for.
        @test issubset((:score, :μ_pred, :σ_pred), propertynames(va))

        # Observed points must be more certain than held-out ones, on average.
        @test mean(va.score[1:25]) < mean(va.score[26:N])
    end
end
