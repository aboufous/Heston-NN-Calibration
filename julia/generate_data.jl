# Étape 0 – génère le jeu d'entraînement -> data/surfaces.npz, et son temps -> results/etape0/generation_time.json.
#
# Usage (depuis la racine du dépôt, avec tous les cœurs) :
#     julia --project=julia --threads=auto julia/generate_data.jl 50000 data/surfaces.npz
#
# X (N, 5)  : [v0, theta, kappa, xi, rho] tirés uniformément dans param_ranges (graine split.seed de base.yaml),
#             v0 et theta uniformément en vol (sample_in_vol), voir src/params.jl ;
# Y (N, 88) : volatilités implicites, maturité par maturité (les 11 strikes de T = 0,1, puis T = 0,3, etc.).
#
# Toutes les surfaces sont GARDÉES, même celles qui contiennent des NaN. Un NaN marque un point où le prix
# de l'option est < 1e-10, donc impossible à inverser en volatilité (strikes 0,5 / 1,5 à T = 0,1 quand la
# variance est faible). Côté Python, ignorer ces points avec np.isnan(Y).

include(joinpath(@__DIR__, "src", "HestonFourier.jl"))
include(joinpath(@__DIR__, "src", "params.jl"))
using .HestonFourier, NPZ, YAML, JSON3, Random

function main(n, out)
    root = dirname(@__DIR__)
    cfg = YAML.load_file(joinpath(root, "configs", "base.yaml"))
    Ts, Ks = Float64.(cfg["grid"]["maturities"]), Float64.(cfg["grid"]["moneyness"])
    seed = cfg["split"]["seed"]

    # tirage AVANT le calcul parallèle : le résultat ne dépend pas du nombre de threads
    X = draw_params(cfg, Xoshiro(seed), n)
    Y = Matrix{Float64}(undef, n, length(Ts) * length(Ks))

    price_surface(X[1, :], Ts, Ks)   # compilation, non chronométrée
    seconds = @elapsed Threads.@threads for i in 1:n
        Y[i, :] = vec(permutedims(price_surface(view(X, i, :), Ts, Ks)))
    end

    mkpath(dirname(abspath(out)))
    npzwrite(out, Dict("X" => X, "Y" => Y))

    n_bad = count(i -> any(isnan, view(Y, i, :)), 1:n)
    timing = Dict("n_surfaces" => n, "n_surfaces_with_nan" => n_bad, "n_nan_points" => count(isnan, Y),
                  "seconds" => seconds, "seconds_per_surface" => seconds / n,
                  "n_threads" => Threads.nthreads(), "machine" => Sys.CPU_NAME)
    mkpath(joinpath(root, "results", "etape0"))
    open(io -> JSON3.pretty(io, timing), joinpath(root, "results", "etape0", "generation_time.json"), "w")

    println("$n surfaces en $(round(seconds; digits = 1)) s -> $out ; ",
            "$n_bad surfaces avec au moins un NaN, $(count(isnan, Y)) points NaN sur $(length(Y)).")
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50_000, length(ARGS) >= 2 ? ARGS[2] : "data/surfaces.npz")
