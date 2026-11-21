# Tirage des paramètres de Heston selon configs/base.yaml. Utilisé par generate_data.jl et benchmark.jl.

const PARAM_NAMES = ("v0", "theta", "kappa", "xi", "rho")

"""
    draw_params(cfg, rng, n) -> Matrix (n, 5)

Tire n jeux [v0, theta, kappa, xi, rho] uniformément dans `param_ranges` :
- les paramètres listés dans `sample_in_vol` (v0 et theta) sont tirés uniformément en vol (racine carrée),
  puis élevés au carré : les plages de base.yaml restent exprimées en variance ;
- les jeux dont le ratio de Feller 2 kappa theta / xi^2 est inférieur à `min_feller_ratio` sont rejetés
  (zone où le pricer COS n'est plus fiable et qui ne correspond pas aux calibrations de marché).
Déterministe pour une graine donnée.
"""
function draw_params(cfg, rng, n)
    ranges = [Float64.(cfg["param_ranges"][k]) for k in PARAM_NAMES]
    in_vol = [k in get(cfg, "sample_in_vol", String[]) for k in PARAM_NAMES]
    min_ratio = Float64(get(cfg, "min_feller_ratio", 0.0))
    lo = [v ? sqrt(r[1]) : r[1] for (r, v) in zip(ranges, in_vol)]
    hi = [v ? sqrt(r[2]) : r[2] for (r, v) in zip(ranges, in_vol)]

    X = Matrix{Float64}(undef, 0, 5)
    while size(X, 1) < n
        B = lo' .+ rand(rng, n, 5) .* (hi .- lo)'
        B[:, in_vol] .= B[:, in_vol] .^ 2
        feller = 2 .* B[:, 3] .* B[:, 2] ./ B[:, 4] .^ 2
        X = vcat(X, B[feller .>= min_ratio, :])
    end
    return X[1:n, :]
end
