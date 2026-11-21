# Étape 0 – temps de Fourier pour UNE surface, sur 1 thread -> results/etape0/fourier_time.json.
#
# Usage :
#     julia --project=julia julia/benchmark.jl
#
# Mêmes jeux de paramètres et même méthode que la mesure Python du notebook 00 (section 4.2), pour que les deux
# temps soient comparables : les N_SETS premières surfaces de data/surfaces.npz ; pour chacune, un appel à blanc
# puis N_REPEATS mesures, dont on garde la médiane ; le temps de référence est la médiane sur les N_SETS jeux.
# Le nombre de termes COS dépend des paramètres, d'où des temps qui varient d'un jeu à l'autre. Lu par le notebook 01.

include(joinpath(@__DIR__, "src", "HestonFourier.jl"))
using .HestonFourier, BenchmarkTools, JSON3, NPZ, YAML, Statistics

const N_SETS = 200     # jeux de paramètres : les 200 premières surfaces de data/surfaces.npz
const N_REPEATS = 5    # mesures par jeu (après un appel à blanc)

function main()
    root = dirname(@__DIR__)
    cfg = YAML.load_file(joinpath(root, "configs", "base.yaml"))
    Ts, Ks = Float64.(cfg["grid"]["maturities"]), Float64.(cfg["grid"]["moneyness"])
    X = npzread(joinpath(root, "data", "surfaces.npz"))["X"]

    times = map(1:N_SETS) do i
        p = X[i, :]
        b = @benchmark price_surface($p, $Ts, $Ks) samples = N_REPEATS evals = 1 seconds = 60
        median(b).time * 1e-9
    end

    result = Dict("median_s" => median(times), "min_s" => minimum(times), "max_s" => maximum(times),
                  "n_parameter_sets" => N_SETS, "n_repeats" => N_REPEATS,
                  "n_threads" => Threads.nthreads(), "machine" => Sys.CPU_NAME)
    mkpath(joinpath(root, "results", "etape0"))
    open(io -> JSON3.pretty(io, result), joinpath(root, "results", "etape0", "fourier_time.json"), "w")
    println("Temps médian de Fourier pour une surface : $(round(1e3 * median(times); digits = 3)) ms " *
            "($N_SETS jeux de paramètres, $N_REPEATS mesures chacun)")
end

main()
