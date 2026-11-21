"""Build the SPX implied-volatility surface on an 8 x 11 grid, using Yahoo Finance.

Usage (from the repository root, with the .venv activated):
    python -m heston_nn.data.market                          # download today's data and build the surface
    python -m heston_nn.data.market --from-raw FILE.csv      # rebuild from a saved raw file
    python -m heston_nn.data.market --force                  # redo today even if it already exists
    python -m heston_nn.data.market --summary                # list the days collected so far

Steps: download -> clean -> forward (put-call parity) -> implied vols -> SVI smile -> grid.
Outputs go to data/spx/ (raw quotes and surfaces, one set of files per day) and results/figures/spx/ (one figure
per day). After each day, data/spx_surface.npz is rebuilt from all
the days collected: it is the file used for the calibration (notebook 04), with NaN where the grid point is not
backed by real quotes.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from heston_nn.config import BASE_CONFIG, load_yaml, repo_path
from heston_nn.data.grid import get_grid, surface_to_vector
from heston_nn.pricing.black_scholes import implied_vol   # Black-76 : une seule version dans le projet

TICKER = "^SPX"
# The grid (maturities in years, strike / forward) comes from configs/base.yaml
GRID_T, GRID_K = get_grid(load_yaml(BASE_CONFIG))

MIN_DAYS = 7              # ignore expiries closer than this
MAX_YEARS = 2.6           # ignore expiries further than this
MAX_REL_SPREAD = 0.50     # drop quotes with (ask - bid) / mid above this
MIN_MID = 0.05            # drop very cheap options
MIN_SMILE_POINTS = 6      # minimum number of vols needed to fit a smile
MAX_SMILE_RMSE = 0.02     # drop a smile if the fit error is bigger than this
FIT_RANGE = (0.4, 1.7)    # only use strikes with K/F in this range
MIN_T_RATE = 0.25         # expiries used to estimate the rate of the day (shorter ones are too noisy)
SLEEP_BETWEEN = 0.5       # seconds to wait between downloads

DEFAULT_OUTDIR = repo_path("data", "spx")                     # raw quotes and surfaces (data), one set per day
CALIBRATION_FILE = repo_path("data", "spx_surface.npz")       # all the days, ready for the calibration
FIGURES_DIR = repo_path("results", "figures", "spx")          # one figure per day (all figures live in results/)


def download_chain(ticker):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError("Could not get the spot price from Yahoo Finance.")
    spot = float(hist["Close"].iloc[-1])

    today = dt.date.today()
    expiries = []
    for e in tk.options:
        days = (dt.date.fromisoformat(e) - today).days
        if MIN_DAYS <= days <= MAX_YEARS * 365:
            expiries.append(e)
    if len(expiries) == 0:
        raise RuntimeError("Yahoo returned no usable expiries.")

    print(f"Spot {spot:.2f}, downloading {len(expiries)} expiries")
    frames = []
    for e in expiries:
        chain = None
        for attempt in range(3):
            try:
                chain = tk.option_chain(e)
                break
            except Exception as error:
                print(f"  problem with {e}: {error}")
                time.sleep(2)
        if chain is None:
            continue

        columns = ["strike", "bid", "ask", "volume", "openInterest"]
        calls = chain.calls[columns].copy()
        calls["type"] = "call"
        puts = chain.puts[columns].copy()
        puts["type"] = "put"
        for df in (calls, puts):
            df["expiry"] = e
            frames.append(df)
        time.sleep(SLEEP_BETWEEN)

    raw = pd.concat(frames, ignore_index=True)
    raw["spot"] = spot
    raw["trade_date"] = today.isoformat()
    raw["download_time"] = dt.datetime.now().isoformat(timespec="seconds")   # to spot a download after the close
    return raw


def clean_quotes(df):
    df = df.copy()
    for col in ["bid", "ask", "volume", "openInterest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[df["mid"] >= MIN_MID]
    df = df[(df["ask"] - df["bid"]) / df["mid"] <= MAX_REL_SPREAD]
    df = df[(df["volume"] > 0) | (df["openInterest"] > 0)]
    return df


def forward_from_parity(calls, puts, spot, D=None):
    """Put-call parity: C - P = D*F - D*K, a straight line in K (10 strikes closest to the money).

    D not given: fit the line, which gives D (slope) and F. Used only to estimate the rate of the day.
    D given (from the rate of the day, see build_surface): F = K + (C - P) / D, median over the same strikes.
    Returns (F, D) or None if the data looks wrong."""
    pairs = calls[["strike", "mid"]].merge(puts[["strike", "mid"]],
                                           on="strike", suffixes=("_c", "_p"))
    if len(pairs) < 3:
        return None

    pairs["diff"] = pairs["mid_c"] - pairs["mid_p"]
    # keep the 10 strikes where call and put prices are closest (near the money)
    near = pairs.loc[pairs["diff"].abs().nsmallest(10).index]
    if len(near) < 3:
        return None

    if D is None:
        slope, intercept = np.polyfit(near["strike"], near["diff"], 1)
        D = -slope
        if not 0.75 < D < 1.05:
            return None
        F = intercept / D
    else:
        F = float(np.median(near["strike"] + near["diff"] / D))
    if not 0.7 * spot < F < 1.3 * spot:
        return None
    return F, D


def day_rate(parity_estimates):
    """One interest rate for the whole day, from the per-expiry estimates [(T, D), ...] of the put-call parity.

    Each expiry alone is noisy (a few strikes, bid-ask spreads): on 2026-11-20 the implied rates went from -36 %
    to +235 %, and a wrong D shifts the forward and the vols of its expiry (about 1 vol point for January 2028).
    The rate is about the same for every maturity, so we take the median over the expiries of at least MIN_T_RATE
    years (all of them if there are fewer than 3), and use D = exp(-rate * T) for every expiry."""
    T, D = np.array(parity_estimates, dtype=float).T
    rates = -np.log(D) / T
    longer = T >= MIN_T_RATE
    return float(np.median(rates[longer] if longer.sum() >= 3 else rates))


def otm_vols(df_exp, F, D, T):
    """Implied vols of out-of-the-money options: puts below F, calls above F."""
    rows = []
    for _, r in df_exp.iterrows():
        is_call = r["type"] == "call"
        if is_call and r["strike"] < F:
            continue
        if (not is_call) and r["strike"] >= F:
            continue
        iv = implied_vol(r["mid"] / D, F, r["strike"], T, is_call)
        if np.isfinite(iv):
            rows.append((r["strike"] / F, iv))
    return pd.DataFrame(rows, columns=["moneyness", "iv"])


def svi_w(k, a, b, rho, m, sig):
    """SVI total variance, k = log(K/F)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def fit_svi(moneyness, iv, T):
    """Fit an SVI smile. Returns (parameters, rmse) or (None, nan)."""
    k = np.log(moneyness)
    w_market = iv ** 2 * T
    w_max = w_market.max()

    def residuals(p):
        a, b, rho, m, sig = p
        w = np.maximum(svi_w(k, *p), 1e-10)
        errors = np.sqrt(w / T) - iv
        # penalty if the smallest variance would be negative
        lowest = a + b * sig * np.sqrt(1 - rho ** 2)
        return np.append(errors, 10 * min(lowest, 0.0))

    lower = [-w_max, 1e-6, -0.999, -1.0, 1e-4]
    upper = [2 * w_max, 5.0, 0.999, 1.0, 2.0]

    best = None
    for rho0 in [-0.7, -0.3, 0.0]:
        for sig0 in [0.05, 0.2]:
            start = [0.5 * w_market.min(), 0.1, rho0, 0.0, sig0]
            try:
                result = least_squares(residuals, start, bounds=(lower, upper))
            except ValueError:
                continue
            if best is None or result.cost < best.cost:
                best = result

    if best is None:
        return None, np.nan
    rmse = np.sqrt(np.mean(best.fun[:-1] ** 2))
    return best.x, rmse


def build_surface(raw):
    trade_date = dt.date.fromisoformat(str(raw["trade_date"].iloc[0]))
    spot = float(raw["spot"].iloc[0])
    quotes = clean_quotes(raw)

    # first pass: one discount factor per expiry (noisy), only to get the rate of the day
    expiries, parity_estimates = [], []
    for expiry, df_exp in quotes.groupby("expiry"):
        T = (dt.date.fromisoformat(expiry) - trade_date).days / 365.0
        if T < MIN_DAYS / 365:
            continue
        calls = df_exp[df_exp["type"] == "call"]
        puts = df_exp[df_exp["type"] == "put"]
        expiries.append((expiry, T, df_exp, calls, puts))
        first = forward_from_parity(calls, puts, spot)
        if first is not None:
            parity_estimates.append((T, first[1]))
    if len(parity_estimates) < 2:
        raise RuntimeError("Fewer than 2 usable expiries. Try again during US market hours "
                           "(bids are often 0 after the close).")
    rate = day_rate(parity_estimates)

    # second pass: the same rate for every expiry, then forward, vols and smile
    smiles = []
    market_points = {}
    for expiry, T, df_exp, calls, puts in expiries:
        result = forward_from_parity(calls, puts, spot, D=np.exp(-rate * T))
        if result is None:
            continue
        F, D = result

        vols = otm_vols(df_exp, F, D, T)
        vols = vols[vols["moneyness"].between(FIT_RANGE[0], FIT_RANGE[1])]
        if len(vols) < MIN_SMILE_POINTS:
            continue

        params, rmse = fit_svi(vols["moneyness"].values, vols["iv"].values, T)
        if params is None or rmse > MAX_SMILE_RMSE:
            continue

        smiles.append({"expiry": expiry, "T": T, "F": F, "D": D,
                       "k_min": vols["moneyness"].min(),
                       "k_max": vols["moneyness"].max(),
                       "rmse": rmse, "a": params[0], "b": params[1],
                       "rho": params[2], "m": params[3], "sig": params[4]})
        market_points[expiry] = vols

    if len(smiles) < 2:
        raise RuntimeError("Fewer than 2 usable expiries. Try again during US market hours "
                           "(bids are often 0 after the close).")
    sm = pd.DataFrame(smiles).sort_values("T").reset_index(drop=True)

    # total variance of each listed expiry at the 11 grid strikes
    log_k = np.log(GRID_K)
    W = []
    covered = []     # True where the grid strike lies inside the quoted range
    for r in sm.itertuples():
        W.append(np.maximum(svi_w(log_k, r.a, r.b, r.rho, r.m, r.sig), 1e-10))
        covered.append((GRID_K >= r.k_min - 0.02) & (GRID_K <= r.k_max + 0.02))
    W = np.array(W)
    covered = np.array(covered)
    Ts = sm["T"].values

    # interpolate total variance in maturity (flat vol outside the listed range)
    surface = np.zeros((len(GRID_T), len(GRID_K)))
    mask = np.zeros((len(GRID_T), len(GRID_K)), dtype=int)
    for i, T in enumerate(GRID_T):
        if T <= Ts[0]:
            w = W[0] * T / Ts[0]
            quoted = covered[0] & (T >= 0.8 * Ts[0])
        elif T >= Ts[-1]:
            w = W[-1] * T / Ts[-1]
            quoted = covered[-1] & (T <= 1.2 * Ts[-1])
        else:
            j = np.searchsorted(Ts, T)
            x = (T - Ts[j - 1]) / (Ts[j] - Ts[j - 1])
            w = (1 - x) * W[j - 1] + x * W[j]
            quoted = covered[j - 1] & covered[j]
        surface[i] = np.sqrt(w / T)
        mask[i] = quoted.astype(int)

    # total variance should grow with maturity, otherwise there is calendar arbitrage
    w_grid = surface ** 2 * GRID_T[:, None]
    n_calendar = int((np.diff(w_grid, axis=0) < -1e-6).sum())

    info = {"trade_date": trade_date.isoformat(), "spot": spot, "rate": rate,
            "n_expiries_used": len(sm), "n_calendar_violations": n_calendar,
            "pct_quoted": 100 * mask.mean()}
    return surface, mask, sm, market_points, info


def plot_surface(surface, mask, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: heatmap of the surface
    image = ax1.imshow(surface * 100, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(len(GRID_K)))
    ax1.set_xticklabels([f"{k:.1f}" for k in GRID_K])
    ax1.set_yticks(range(len(GRID_T)))
    ax1.set_yticklabels([f"{t:.1f}" for t in GRID_T])
    ax1.set_xlabel("K / F")
    ax1.set_ylabel("Maturity (years)")
    ax1.set_title("Implied vol (%)")
    fig.colorbar(image, ax=ax1)

    # right: a few smiles, dashed where the values are extrapolated
    for i in [0, 2, 4, 7]:
        ax2.plot(GRID_K, surface[i] * 100, "--", alpha=0.4)
        quoted = mask[i] == 1
        ax2.plot(GRID_K[quoted], surface[i][quoted] * 100,
                 label=f"T = {GRID_T[i]:.1f}", color=ax2.lines[-1].get_color())
    ax2.set_xlabel("K / F")
    ax2.set_ylabel("Implied vol (%)")
    ax2.set_title("Smiles (dashed = extrapolated)")
    ax2.legend()

    fig.suptitle(f"SPX surface {info['trade_date']} (spot {info['spot']:.0f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_outputs(surface, mask, sm, info, outdir, figures_dir=FIGURES_DIR):
    date = info["trade_date"]
    (outdir / "surfaces").mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    columns = [f"{k:.1f}" for k in GRID_K]
    index = [f"{t:.1f}" for t in GRID_T]
    pd.DataFrame(surface, index=index, columns=columns).to_csv(
        outdir / "surfaces" / f"{date}_surface.csv")
    pd.DataFrame(mask, index=index, columns=columns).to_csv(
        outdir / "surfaces" / f"{date}_mask.csv")
    sm.to_csv(outdir / "surfaces" / f"{date}_smiles.csv", index=False)
    plot_surface(surface, mask, info, figures_dir / f"{date}_surface.png")

    # one row per day with the 88 vols (T outer loop, K inner loop)
    names = [f"T{t:.1f}_K{k:.1f}" for t in GRID_T for k in GRID_K]
    vols = surface_to_vector(surface).round(6)
    quoted = surface_to_vector(mask)
    row = dict(info)
    for name, vol, q in zip(names, vols, quoted):
        row[f"iv_{name}"] = vol
        row[f"q_{name}"] = int(q)

    wide_path = outdir / "surfaces_wide.csv"
    new = pd.DataFrame([row])
    if wide_path.exists():
        old = pd.read_csv(wide_path)
        old = old[old["trade_date"] != date]      # replace the day if re-run
        new = pd.concat([old, new], ignore_index=True).sort_values("trade_date")
    new.to_csv(wide_path, index=False)


def save_for_calibration(outdir=DEFAULT_OUTDIR, path=CALIBRATION_FILE):
    """Rebuild the calibration file from all the days in outdir/surfaces_wide.csv.

    Arrays in the .npz file (vols in decimal, 0.20 = 20 %):
    - surface (8, 11): the most recent day, NaN where the point is not backed by real quotes (mask = 0);
    - surfaces (n_days, 8, 11): every day, same NaN rule; dates (n_days,) and spots (n_days,);
    - surfaces_full, masks (n_days, 8, 11): the full SVI surface (extrapolated points included) and the mask;
    - maturities (8,), moneyness (11,): the grid.
    We calibrate only on quoted points: extrapolated values come from the SVI fit, not from prices.
    """
    wide = pd.read_csv(Path(outdir) / "surfaces_wide.csv").sort_values("trade_date")
    names = [f"T{t:.1f}_K{k:.1f}" for t in GRID_T for k in GRID_K]
    shape = (len(wide), len(GRID_T), len(GRID_K))
    full = wide[[f"iv_{name}" for name in names]].to_numpy(dtype=float).reshape(shape)
    masks = wide[[f"q_{name}" for name in names]].to_numpy(dtype=int).reshape(shape)
    quoted = np.where(masks == 1, full, np.nan)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, surface=quoted[-1], surfaces=quoted, surfaces_full=full, masks=masks,
             dates=wide["trade_date"].to_numpy(dtype=str), spots=wide["spot"].to_numpy(dtype=float),
             maturities=GRID_T, moneyness=GRID_K)
    return path


def collect(outdir=DEFAULT_OUTDIR, ticker=TICKER, from_raw=None, force=False, calibration_file=CALIBRATION_FILE,
            figures_dir=FIGURES_DIR):
    """Download today's quotes (or read a saved raw file), build the surface, save everything
    (files of the day in outdir, then the calibration file rebuilt from all the days).

    Returns (surface, mask, smiles, info), or None if nothing was done (weekend, or today already collected).
    """
    outdir = Path(outdir)
    if from_raw is not None:
        raw = pd.read_csv(from_raw)
    else:
        today = dt.date.today()
        if today.weekday() >= 5:
            print("It's the weekend, Yahoo would return stale quotes. Try on a weekday.")
            return None
        raw_path = outdir / "raw" / f"{today.isoformat()}_SPX_chain.csv"
        if raw_path.exists() and not force:
            print(f"Today's data already exists ({raw_path}). Use force=True (--force) to redo.")
            return None
        raw = download_chain(ticker)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_path, index=False)
        print(f"Raw quotes saved: {raw_path} ({len(raw)} rows)")

    surface, mask, sm, market_points, info = build_surface(raw)
    save_outputs(surface, mask, sm, info, outdir, figures_dir)
    save_for_calibration(outdir, calibration_file)
    return surface, mask, sm, info


def summary(outdir):
    wide_path = outdir / "surfaces_wide.csv"
    if not wide_path.exists():
        print("No surfaces collected yet.")
        return
    df = pd.read_csv(wide_path)
    print(df[["trade_date", "spot", "n_expiries_used", "pct_quoted",
              "n_calendar_violations"]].to_string(index=False))
    print(f"\n{len(df)} day(s) collected.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--from-raw", help="rebuild from a saved raw CSV")
    parser.add_argument("--force", action="store_true", help="redo today if it exists")
    parser.add_argument("--summary", action="store_true", help="list collected days")
    args = parser.parse_args()
    outdir = Path(args.outdir)

    if args.summary:
        summary(outdir)
        return

    result = collect(outdir, args.ticker, from_raw=args.from_raw, force=args.force)
    if result is None:
        sys.exit(0)
    surface, mask, sm, info = result

    print(f"\nSurface for {info['trade_date']}: {info['n_expiries_used']} expiries used, "
          f"{info['pct_quoted']:.0f}% of the grid backed by quotes.")
    table = pd.DataFrame(surface * 100, index=[f"T={t}" for t in GRID_T],
                         columns=[f"{k:.1f}" for k in GRID_K])
    print(table.round(1).to_string())
    print(f"Calibration file updated: {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
