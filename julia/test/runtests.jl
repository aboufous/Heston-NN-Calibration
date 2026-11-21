# Tests du pricer.  Usage : julia --project=julia julia/test/runtests.jl

include(joinpath(@__DIR__, "..", "src", "HestonFourier.jl"))
include(joinpath(@__DIR__, "..", "src", "params.jl"))
using .HestonFourier, Test, YAML, Random

const CFG = YAML.load_file(joinpath(@__DIR__, "..", "..", "configs", "base.yaml"))
const Ts = Float64.(CFG["grid"]["maturities"])
const Ks = Float64.(CFG["grid"]["moneyness"])

@testset "HestonFourier" begin
    # prix de référence de Fang & Oosterlee (2008) : call à la monnaie, T = 1, S0 = K = 100
    @test 100 * otm_prices([1.0], 1.0, [0.0175, 0.0398, 1.5768, 0.5751, -0.5711])[1] ≈ 5.785155435 atol = 1e-7

    # Black-76 : implied_vol inverse black_otm ; NaN pour un prix impossible
    for K in Ks, T in Ts, σ in (0.05, 0.2, 0.6)
        price = black_otm(K, T, σ)
        price > 1e-10 && @test implied_vol(price, K, T) ≈ σ atol = 1e-6
    end
    @test isnan(implied_vol(-0.1, 1.0, 1.0)) && isnan(implied_vol(2.0, 1.0, 1.0))

    # limite Black–Scholes (xi petit, rho = 0, v0 = theta) : surface plate à sqrt(v0) = 0,2
    s = price_surface([0.04, 0.04, 2.0, 0.01, 0.0], Ts, Ks)
    @test count(!isnan, s) >= 80
    @test maximum(abs.(filter(!isnan, s) .- 0.2)) < 1e-3

    # pas d'arbitrage : prix du call (parité pour K < 1) décroissant et convexe en K
    for p in ([0.04, 0.06, 1.5, 0.5, -0.7], [0.2, 0.1, 5.0, 0.9, -0.3], [0.02, 0.2, 8.0, 0.3, 0.0]), T in Ts
        C = otm_prices(Ks, T, p) .+ max.(1 .- Ks, 0)
        @test all(diff(C) .<= 1e-12) && all(diff(diff(C)) .>= -1e-10)
    end

    # cumulants : variance positive, et égale à v0 * T dans la limite Black–Scholes (xi petit, v0 = theta)
    @test HestonFourier.cumulants(1.0, [0.04, 0.04, 2.0, 0.01, 0.0])[2] ≈ 0.04 atol = 1e-3
    for p in ([0.006, 0.216, 0.819, 1.122, -0.783], [0.027, 0.179, 0.571, 1.001, -0.121]), T in Ts
        @test HestonFourier.cumulants(T, p)[2] > 0     # l'ancienne formule fermée donnait < 0 ici
    end

    # Feller très violée (queue gauche épaisse) : références indépendantes (formule de Lewis, intégration
    # numérique). Avec l'ancien intervalle (L = 12, formule fermée de c2) : erreurs de 178 et 22 pb.
    @test price_surface([0.0332, 0.0197, 1.9677, 1.151, -0.9054], [1.2], [1.5])[1] ≈ 0.082825 atol = 1e-5
    @test price_surface([0.0191, 0.0726, 0.95, 1.326, -0.467], [1.2], [0.5])[1] ≈ 0.351051 atol = 1e-5
    # vol élevée, maturité longue : sans le plafond B_MAX de la borne droite, e^b ≈ 1e9 faisait perdre ~0,5 pb
    @test price_surface([0.135, 0.1415, 1.0408, 1.3088, -0.9138], [2.0], [1.5])[1] ≈ 0.133554 atol = 1e-5

    # tirage des paramètres : dans les plages de base.yaml, ratio de Feller respecté, reproductible
    X = draw_params(CFG, Xoshiro(1), 1000)
    @test size(X) == (1000, 5)
    for (j, k) in enumerate(PARAM_NAMES)
        lo, hi = CFG["param_ranges"][k]
        @test all(lo - 1e-12 .<= X[:, j] .<= hi + 1e-12)
    end
    @test all(2 .* X[:, 3] .* X[:, 2] ./ X[:, 4] .^ 2 .>= CFG["min_feller_ratio"])
    @test X == draw_params(CFG, Xoshiro(1), 1000)

    # surface typique : forme (8, 11), NaN seulement à T = 0,1, valeurs plausibles, skew négatif si rho < 0
    s = price_surface([0.04, 0.06, 1.5, 0.5, -0.7], Ts, Ks)
    @test size(s) == (8, 11)
    @test !any(isnan, s[2:end, :])
    @test all(0.01 .< filter(!isnan, s) .< 2)
    @test all(s[:, 5] .> s[:, 7])
end
