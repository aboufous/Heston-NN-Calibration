import numpy as np


def get_grid(cfg):
    """Return (maturities (8,), moneyness (11,)) from the config."""
    maturities = np.array(cfg["grid"]["maturities"], dtype=float)
    moneyness = np.array(cfg["grid"]["moneyness"], dtype=float)
    return maturities, moneyness


def surface_to_vector(surface):
    """(8, 11) -> (88,), maturity by maturity."""
    surface = np.asarray(surface)
    if surface.shape != (8, 11):
        raise ValueError(f"Expected shape (8, 11), got {surface.shape}")
    return surface.reshape(88)


def vector_to_surface(vector):
    """(88,) -> (8, 11)."""
    vector = np.asarray(vector)
    if vector.shape != (88,):
        raise ValueError(f"Expected shape (88,), got {vector.shape}")
    return vector.reshape(8, 11)


def calibration_zone(cfg):
    """(8, 11) booleans: True for the grid points used to calibrate on the SPX (cfg["spx_calibration"]).

    Usage: target = np.where(calibration_zone(cfg), spx_surface, np.nan) ; the calibration ignores NaN points.
    """
    maturities, moneyness = get_grid(cfg)
    zone = cfg["spx_calibration"]
    low, high = zone["moneyness"]
    in_strikes = (moneyness >= low - 1e-9) & (moneyness <= high + 1e-9)
    in_maturities = maturities >= zone["min_maturity"] - 1e-9
    return in_maturities[:, None] & in_strikes[None, :]
