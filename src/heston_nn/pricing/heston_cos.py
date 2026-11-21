"""Pricer de Heston par la méthode COS, en Python : copie fidèle de julia/src/HestonFourier.jl.

À quoi il sert : comparer Python et Julia (notebook 00) sur le temps de génération et vérifier qu'ils donnent
les mêmes surfaces. Les données d'entraînement et le temps de référence de Fourier restent ceux de Julia ;
ce module peut aussi servir de pricer Fourier côté Python (par exemple pour la calibration de l'étape 4).

Mêmes conventions et mêmes réglages que le Julia (voir son en-tête pour les justifications) :
F = 1, taux nul, K = moneyness, params = [v0, theta, kappa, xi, rho] ; fonction caractéristique « little
Heston trap » ; intervalle c1 ± 32 √c2 (borne droite plafonnée à 10) avec les cumulants calculés à partir de la
fonction caractéristique ;
option hors de la monnaie ; NaN si le prix est < 1e-10.
Seule différence : l'inversion de Black-76 se fait par dichotomie vectorisée (les 11 strikes d'un coup) au
lieu de la méthode de Brent ; les deux trouvent la même racine, à 1e-15 près.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from heston_nn.pricing.black_scholes import black76

N_MIN, N_MAX = 256, 32768     # nombre de termes de la série COS
CF_TOL = 1e-15                # on double le nombre de termes tant que |φ(u_max)| > CF_TOL
L = 32.0                      # largeur de l'intervalle de troncature, en écarts-types
MARGIN = 1.5                  # marge (en log) autour des strikes extrêmes
B_MAX = 10.0                  # borne droite max (sinon e^b fait perdre de la précision aux calls)
MIN_PRICE = 1e-10             # en dessous, le prix n'est plus assez précis pour être inversé -> NaN
VOL_LO, VOL_HI = 1e-4, 5.0    # intervalle de recherche de la volatilité implicite
N_BISECTION = 60              # (VOL_HI - VOL_LO) / 2^60 ≈ 4e-18

PARAM_NAMES = ("v0", "theta", "kappa", "xi", "rho")


def cf(u, T, p):
    """Fonction caractéristique de log(F_T / F) sous Heston (little Heston trap). u : réel ou tableau."""
    v0, theta, kappa, xi, rho = p
    u = np.asarray(u, dtype=complex)
    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta**2 + xi**2 * (1j * u + u**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((beta - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (beta - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def cumulants(T, p, h=1e-4):
    """Cumulants c1 (moyenne) et c2 (variance) de log(F_T / F), par différences finies de log φ en 0."""
    lp, lm = np.log(cf(h, T, p)), np.log(cf(-h, T, p))
    c1 = (lp - lm).imag / (2 * h)
    c2 = -(lp + lm).real / h**2          # log φ(0) = 0
    return c1, c2


def cos_range(T, p, strikes):
    """Intervalle [a, b] : c1 ± L √c2 (b plafonné à B_MAX), élargi pour contenir tous les log-strikes."""
    c1, c2 = cumulants(T, p)
    sd = np.sqrt(max(c2, 1e-8))
    a = min(c1 - L * sd, np.log(np.min(strikes)) - MARGIN)
    b = max(min(c1 + L * sd, B_MAX), np.log(np.max(strikes)) + MARGIN)
    return a, b


def otm_prices(strikes, T, p):
    """Prix des options hors de la monnaie (put si K < 1, call sinon) pour une maturité, par la méthode COS."""
    K = np.asarray(strikes, dtype=float)
    a, b = cos_range(T, p, K)
    N = N_MIN
    while N < N_MAX and abs(cf(N * np.pi / (b - a), T, p)) > CF_TOL:
        N *= 2
    u = np.arange(N) * np.pi / (b - a)
    A = 2 / (b - a) * (cf(u, T, p) * np.exp(-1j * u * a)).real
    A[0] /= 2                                            # premier terme de la série divisé par 2

    w = u[:, None]                                       # (N, 1) contre (1, n_strikes)
    k = np.clip(np.log(K), a, b)[None, :]

    def psi(c, d):                                       # ∫_c^d cos(w (z - a)) dz
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (np.sin(w * (d - a)) - np.sin(w * (c - a))) / w
        r[0] = d - c                                     # w = 0
        return r

    def chi(c, d):                                       # ∫_c^d e^z cos(w (z - a)) dz
        return (np.exp(d) * (np.cos(w * (d - a)) + w * np.sin(w * (d - a)))
                - np.exp(c) * (np.cos(w * (c - a)) + w * np.sin(w * (c - a)))) / (1 + w**2)

    put = A @ (K * psi(a, k) - chi(a, k))
    call = A @ (chi(k, b) - K * psi(k, b))
    return np.maximum(np.where(K < 1, put, call), 0.0)


def black_otm(K, T, sigma):
    """Prix Black-76 (F = 1, taux nul) de l'option hors de la monnaie : put si K < 1, call sinon."""
    K = np.asarray(K, dtype=float)
    return black76(1.0, K, T, sigma, K >= 1)


def implied_vol(prices, strikes, T):
    """Volatilités implicites (vectorisé) ; NaN si le prix est hors de ]MIN_PRICE, min(K, 1)[ ou hors de l'intervalle."""
    price, K = np.broadcast_arrays(np.asarray(prices, float), np.asarray(strikes, float))
    lo, hi = np.full(price.shape, VOL_LO), np.full(price.shape, VOL_HI)
    ok = (price > MIN_PRICE) & (price < np.minimum(K, 1.0))                 # bornes de non-arbitrage
    ok &= (black_otm(K, T, lo) < price) & (black_otm(K, T, hi) > price)      # racine dans l'intervalle
    for _ in range(N_BISECTION):
        mid = 0.5 * (lo + hi)
        below = black_otm(K, T, mid) < price
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def price_surface(params, maturities, moneyness):
    """CONTRAT : params (5,) -> surface (n_maturités, n_strikes) de volatilités implicites, NaN si non fiable."""
    K = np.asarray(moneyness, dtype=float)
    return np.vstack([implied_vol(otm_prices(K, T, params), K, T) for T in maturities])


def draw_params(cfg, rng, n):
    """Tire n jeux de paramètres selon base.yaml, comme julia/src/params.jl (mais avec le générateur de numpy :
    mêmes lois, valeurs différentes). v0 et theta uniformes en vol ; ratio de Feller >= min_feller_ratio."""
    ranges = np.array([cfg["param_ranges"][k] for k in PARAM_NAMES], dtype=float)
    in_vol = np.array([k in cfg.get("sample_in_vol", []) for k in PARAM_NAMES])
    min_ratio = float(cfg.get("min_feller_ratio", 0.0))
    lo, hi = ranges[:, 0].copy(), ranges[:, 1].copy()
    lo[in_vol], hi[in_vol] = np.sqrt(lo[in_vol]), np.sqrt(hi[in_vol])   # racine seulement pour v0 et theta
    kept = []
    while sum(len(B) for B in kept) < n:
        B = lo + rng.random((n, 5)) * (hi - lo)
        B[:, in_vol] = B[:, in_vol] ** 2
        feller = 2 * B[:, 2] * B[:, 1] / B[:, 3] ** 2
        kept.append(B[feller >= min_ratio])
    return np.vstack(kept)[:n]


def _price_rows(args):
    X, maturities, moneyness = args
    return np.array([price_surface(p, maturities, moneyness).ravel() for p in X])


def generate_surfaces(X, maturities, moneyness, n_jobs=1):
    """Y (N, 88) pour les paramètres X (N, 5), maturité par maturité comme le fichier Julia.

    n_jobs > 1 : calcul réparti sur n_jobs processus (l'équivalent des threads de Julia).
    """
    X = np.asarray(X, dtype=float)
    if n_jobs == 1:
        return _price_rows((X, maturities, moneyness))
    chunks = [(c, maturities, moneyness) for c in np.array_split(X, 4 * n_jobs) if len(c)]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return np.vstack(list(pool.map(_price_rows, chunks)))
