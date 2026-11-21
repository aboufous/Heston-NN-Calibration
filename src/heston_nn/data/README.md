# `data/` folder

This folder prepares the data used by the neural network.
The network learns to map **5 Heston parameters** to a **volatility surface of 88 numbers**
(8 maturities x 11 strikes).

| File | What it does |
|------|--------------|
| `market.py` | Builds the **real** SPX volatility surface from market option prices (explained below) |
| `grid.py` | Defines the 8 x 11 grid and converts a surface (8, 11) to a vector (88,) and back |
| `dataset.py` | Loads the synthetic dataset made by Julia and checks that it is valid |
| `split.py` | Splits data into train / validation / test and standardizes it |
| `__init__.py` | Makes the folder a Python package |

---

## `market.py` - real SPX volatility surface

### The idea in one minute

An **option** is a contract whose price depends on the market's expectation of future movement.
**Implied volatility** is the number you put in the Black formula so that it gives the observed option price.
Plotting implied volatility against strike and maturity gives a **surface**.
`market.py` builds this surface from live prices, on the same 8 x 11 grid as the synthetic Heston data,
so the two can be compared.

- Maturities `T` (years): `0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0`
- Strikes `K/F` (strike divided by forward): `0.5, 0.6, ..., 1.5`

### How to run

From the repository root, with the `.venv` activated:

```powershell
python -m heston_nn.data.market                       # download today's data and build the surface
python -m heston_nn.data.market --force               # redo today even if it already exists
python -m heston_nn.data.market --from-raw data/spx/raw/2026-11-20_SPX_chain.csv   # rebuild from a saved file
python -m heston_nn.data.market --summary             # list the days collected so far
```

Or from a notebook: `market.collect()` (notebook 00, section 5).

Tip: run it on a weekday during US market hours (15:30-22:00 Paris time), ideally 16:30-21:30
(Yahoo quotes are delayed by 15 minutes, and the book is thinner right after the open and before the close).
After the close many bids are 0 and the data is unusable. Run it once per day to collect several days.

### The steps, in order

Each step is one function in the file.

1. **Download** - `download_chain`
   Gets all SPX option quotes (bid, ask, strike, expiry) from Yahoo Finance.
   Expiries that are too close (< 7 days) or too far (> 2.6 years) are skipped.
   The raw data is saved in `data/raw/` so you can rebuild later.

2. **Clean** - `clean_quotes`
   Computes the mid price `(bid + ask) / 2` and removes bad quotes:
   no bid, very cheap options, huge bid-ask spread, options nobody trades.

3. **Find the forward** - `forward_from_parity`, `day_rate`
   The forward `F` is the "fair" future price of the index.
   It is found from put-call parity: `Call - Put = D * (F - K)`.
   This is a straight line in `K`, so a line fit gives `F` and the discount factor `D` of each expiry.
   These per-expiry `D` are noisy (on 2026-11-20 the implied rates went from -36 % to +235 %), so we keep only
   **one rate for the day** (median over the expiries of at least 3 months), use `D = exp(-rate * T)` for every
   expiry, and recompute each forward with it: `F = K + (C - P) / D`.

4. **Compute implied volatilities** - `otm_vols` (uses `black76` and `implied_vol` from `heston_nn/pricing/black_scholes.py`)
   For each option we search for the volatility that makes the Black-76 price equal to the market price.
   We only keep **out-of-the-money** options (puts below `F`, calls above `F`), because they are the most reliable.

5. **Fit a smile** - `svi_w`, `fit_svi`
   For each expiry, the vols form a curve (a "smile") with gaps.
   We fit a smooth 5-parameter curve called **SVI** through them.
   If the fit is poor (error above 2 vol points) the expiry is dropped.

6. **Build the grid** - `build_surface`
   Reads each smile at the 11 grid strikes, then interpolates between expiries to get the 8 grid maturities.
   It also returns a **mask**: `1` = backed by real quotes, `0` = extrapolated.
   Finally it checks for calendar arbitrage (total variance `vol^2 * T` should increase with `T`).

7. **Save** - `plot_surface`, `save_outputs`, `save_for_calibration`
   Writes the files of the day in `data/spx/` and its figure in `results/figures/spx/`, then rebuilds
   `data/spx_surface.npz` from all the days collected.

`collect` runs the steps above (download or `--from-raw`, build, save); `main` is the command-line wrapper.

### Output files

In `data/spx/` (versioned on GitHub: a past day cannot be downloaded again). The figure of each day
(heatmap and smiles) goes to `results/figures/spx/DATE_surface.png`, with all the other figures.

| File | Content |
|------|---------|
| `raw/DATE_SPX_chain.csv` | All raw quotes downloaded that day (keep these!) |
| `surfaces/DATE_surface.csv` | The 8 x 11 implied vols |
| `surfaces/DATE_mask.csv` | 1 = real quotes, 0 = extrapolated |
| `surfaces/DATE_smiles.csv` | SVI fit and fit quality for each expiry |
| `surfaces_wide.csv` | One row per day: 88 vols (+ 88 mask columns), same order as the network outputs |

And `data/spx_surface.npz`, used by the calibration (notebook 04), rebuilt after each day. Vols in decimal:

| Array | Shape | Content |
|-------|-------|---------|
| `surface` | (8, 11) | Most recent day, **NaN where the point is not backed by real quotes** |
| `surfaces` | (n_days, 8, 11) | Every day, same NaN rule |
| `dates`, `spots` | (n_days,) | Trade date and spot of each day |
| `surfaces_full`, `masks` | (n_days, 8, 11) | Full SVI surface (extrapolated points included) and mask |
| `maturities`, `moneyness` | (8,), (11,) | The grid |

We calibrate only on quoted points: extrapolated values come from the SVI fit, not from prices (and the Fourier
pricer returns NaN for options too far out of the money, which the calibration would count as a huge error).

**Precision of the pipeline** (test with a synthetic option chain generated by Heston, `tests/test_market.py`):
the forward and the discount factor are recovered exactly; on quoted points the surface is within about 6 bp of
the true one on average, up to about 35 bp at short maturities, because an SVI smile cannot follow a Heston smile
exactly. Keep this in mind when comparing calibration errors.

### Settings you can change

They are at the top of `market.py`: `GRID_T`, `GRID_K`, `MIN_DAYS`, `MAX_YEARS`,
`MAX_REL_SPREAD`, `MIN_MID`, `MIN_SMILE_POINTS`, `MAX_SMILE_RMSE`, `FIT_RANGE`.

---

## `grid.py`

_To be written: what the file does, its main functions, and how to use them._

## `dataset.py`

_To be written: what the file does, its main functions, and how to use them._

## `split.py`

_To be written: what the file does, its main functions, and how to use them._

---

## Adding a new script

Copy this template at the end of the file:

```markdown
## `name.py`

**Purpose:** one sentence.

**Main functions:**
- `function_name(args)` - what it does.

**How to run / use it:** example.

**Inputs / outputs:** what it reads and what it writes.
```
