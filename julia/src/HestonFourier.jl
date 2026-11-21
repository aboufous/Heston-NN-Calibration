"""
HestonFourier : surface de volatilités implicites du modèle de Heston par la méthode de Fourier (COS).

Conventions (identiques au côté Python) : prix forward F = 1, taux nul, strike K = moneyness,
params = [v0, theta, kappa, xi, rho].

- fonction caractéristique « little Heston trap » (Albrecher et al., 2007), stable numériquement ;
- méthode COS (Fang & Oosterlee, 2008) : la densité est développée UNE fois par maturité, puis chaque
  strike ne coûte qu'une somme ;
- intervalle de troncature c1 ± L √c2, avec c1, c2 (moyenne, variance de log F_T) calculés à partir de la
  fonction caractéristique, et L = 32 : quand la condition de Feller est très violée, la queue gauche est
  épaisse et L = 12 donnait des erreurs jusqu'à plusieurs centaines de pb (vérifié contre la formule de Lewis) ;
  la borne droite est plafonnée à 10 (sinon e^b, jusqu'à 1e9, fait perdre jusqu'à 0,8 pb aux calls) ;
- on price l'option HORS de la monnaie (put si K < 1, call sinon) : prix petit mais calculé directement ;
- inversion de Black-76 -> volatilité implicite, NaN si le prix est < 1e-10 : en dessous, l'erreur numérique de
  la méthode COS (quelques 1e-12) fausse la volatilité de plusieurs pb (cas des strikes extrêmes à maturité
  courte et faible variance).
"""
module HestonFourier

using Distributions: Normal, cdf
using Roots: find_zero, Brent

export otm_prices, black_otm, implied_vol, price_surface

const N_MIN, N_MAX = 256, 32768   # nombre de termes de la série COS
const CF_TOL = 1e-15              # on double le nombre de termes tant que |φ(u_max)| > CF_TOL
const L = 32.0                    # largeur de l'intervalle de troncature, en écarts-types (voir en-tête)
const MARGIN = 1.5                # marge (en log) autour des strikes extrêmes
const B_MAX = 10.0                # borne droite max : au-delà, e^b fait perdre de la précision aux calls (et
                                  # un log-rendement > 10, soit F_T > 22 000 F, n'a aucune probabilité)
const MIN_PRICE = 1e-10           # en dessous, le prix n'est plus assez précis pour être inversé -> NaN
const N01 = Normal()

"""Fonction caractéristique de log(F_T / F) sous Heston (little Heston trap)."""
function cf(u, T, p)
    v0, theta, kappa, xi, rho = p
    beta = kappa - rho * xi * im * u
    d = sqrt(beta^2 + xi^2 * (im * u + u^2))
    g = (beta - d) / (beta + d)
    e = exp(-d * T)
    C = kappa * theta / xi^2 * ((beta - d) * T - 2 * log((1 - g * e) / (1 - g)))
    D = (beta - d) / xi^2 * (1 - e) / (1 - g * e)
    return exp(C + D * v0)
end

"""
Cumulants c1 (moyenne) et c2 (variance) de log(F_T / F), par différences finies de log φ en 0 :
log φ(u) ≈ i c1 u - c2 u² / 2. (La formule fermée de c2 de Fang & Oosterlee, utilisée avant, donnait une
variance NÉGATIVE pour environ 20 % des paramètres tirés.)
"""
function cumulants(T, p; h = 1e-4)
    lp, lm = log(cf(h, T, p)), log(cf(-h, T, p))
    c1 = imag(lp - lm) / (2h)
    c2 = -real(lp + lm) / h^2          # log φ(0) = 0
    return c1, c2
end

"""Intervalle [a, b] : c1 ± L √c2 (b plafonné à B_MAX), élargi pour contenir tous les log-strikes."""
function cos_range(T, p, strikes)
    c1, c2 = cumulants(T, p)
    sd = sqrt(max(c2, 1e-8))           # garde-fou si c2 est minuscule
    a = min(c1 - L * sd, log(minimum(strikes)) - MARGIN)
    b = max(min(c1 + L * sd, B_MAX), log(maximum(strikes)) + MARGIN)
    return a, b
end

"""Prix des options hors de la monnaie (put si K < 1, call sinon) pour tous les strikes d'une maturité."""
function otm_prices(strikes, T, p)
    a, b = cos_range(T, p, strikes)
    # maturités courtes : φ décroît lentement, il faut plus de termes
    N = N_MIN
    while N < N_MAX && abs(cf(N * pi / (b - a), T, p)) > CF_TOL
        N *= 2
    end
    u = [k * pi / (b - a) for k in 0:N-1]
    A = [2 / (b - a) * real(cf(uk, T, p) * exp(-im * uk * a)) for uk in u]
    A[1] /= 2
    # ∫_c^d cos(u (z - a)) dz  et  ∫_c^d e^z cos(u (z - a)) dz
    psi(w, c, d) = w == 0 ? d - c : (sin(w * (d - a)) - sin(w * (c - a))) / w
    chi(w, c, d) = (exp(d) * (cos(w * (d - a)) + w * sin(w * (d - a)))
                    - exp(c) * (cos(w * (c - a)) + w * sin(w * (c - a)))) / (1 + w^2)
    return map(strikes) do K
        k = clamp(log(K), a, b)
        s = K < 1 ? sum(A[i] * (K * psi(u[i], a, k) - chi(u[i], a, k)) for i in eachindex(u)) :   # put
                    sum(A[i] * (chi(u[i], k, b) - K * psi(u[i], k, b)) for i in eachindex(u))     # call
        max(s, 0.0)
    end
end

"""Prix Black-76 de l'option hors de la monnaie (put si K < 1, call sinon), F = 1."""
function black_otm(K, T, sigma)
    s = sigma * sqrt(T)
    d1 = -log(K) / s + s / 2
    d2 = d1 - s
    return K < 1 ? K * cdf(N01, -d2) - cdf(N01, -d1) : cdf(N01, d1) - K * cdf(N01, d2)
end

"""Volatilité implicite à partir du prix hors de la monnaie ; NaN si le prix ne peut pas être inversé."""
function implied_vol(price, K, T)
    MIN_PRICE < price < min(K, 1.0) || return NaN          # bornes de non-arbitrage
    f(s) = black_otm(K, T, s) - price
    f(1e-4) < 0 < f(5.0) || return NaN
    return find_zero(f, (1e-4, 5.0), Brent(); xatol = 1e-12)
end

"""CONTRAT : params (5,) -> surface (8, 11) ; lignes = maturités, colonnes = moneyness (ordre de base.yaml)."""
function price_surface(p, maturities, moneyness)
    surface = Matrix{Float64}(undef, length(maturities), length(moneyness))
    for (i, T) in enumerate(maturities)
        surface[i, :] = implied_vol.(otm_prices(moneyness, T, p), moneyness, T)
    end
    return surface
end

end # module
