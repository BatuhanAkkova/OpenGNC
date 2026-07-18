"""
Link budget calculations and RF analysis tools.
"""

import numpy as np

C = 299792458.0


def calculate_friis_link_budget(
    p_tx_w: float,
    g_tx_db: float,
    g_rx_db: float,
    frequency_hz: float,
    distance_m: float,
    losses_misc_db: float = 0.0,
    l_atm_db: float = 0.0,
) -> dict[str, float]:
    """
    Calculate a link budget using the Friis transmission equation.

    Returns
    -------
    dict[str, float]
        Received power and free-space loss terms in linear and dB form.
    """
    if p_tx_w <= 0:
        raise ValueError("Transmitter power must be strictly positive.")
    if distance_m <= 0:
        raise ValueError("Distance must be strictly positive.")
    if frequency_hz <= 0:
        raise ValueError("Frequency must be strictly positive.")

    p_tx_dbw = 10.0 * np.log10(p_tx_w)
    l_fs_db = 20.0 * np.log10(distance_m) + 20.0 * np.log10(frequency_hz) - 147.554
    p_rx_dbw = p_tx_dbw + g_tx_db + g_rx_db - l_fs_db - l_atm_db - losses_misc_db
    p_rx_w = 10.0 ** (p_rx_dbw / 10.0)

    return {
        "p_rx_dbw": float(p_rx_dbw),
        "p_rx_w": float(p_rx_w),
        "l_fs_db": float(l_fs_db),
    }


def calculate_doppler_shift(
    f_tx_hz: float,
    r_ecef_rx: np.ndarray,
    v_ecef_rx: np.ndarray,
    r_ecef_tx: np.ndarray,
    v_ecef_tx: np.ndarray,
) -> dict[str, float]:
    """
    Calculate the Doppler shift for a signal sent from transmitter to receiver.

    Returns
    -------
    dict[str, float]
        Received frequency and Doppler shift in hertz.
    """
    r_rx = np.asarray(r_ecef_rx, dtype=float)
    v_rx = np.asarray(v_ecef_rx, dtype=float)
    r_tx = np.asarray(r_ecef_tx, dtype=float)
    v_tx = np.asarray(v_ecef_tx, dtype=float)

    if r_rx.shape != (3,) or v_rx.shape != (3,) or r_tx.shape != (3,) or v_tx.shape != (3,):
        raise ValueError("Positions and velocities must be vectors of length 3.")

    r_rel = r_rx - r_tx
    v_rel_vec = v_rx - v_tx
    distance = np.linalg.norm(r_rel)
    if distance == 0:
        return {"f_rx_hz": f_tx_hz, "doppler_shift_hz": 0.0}

    range_rate = float(np.dot(r_rel, v_rel_vec) / distance)
    doppler_shift_hz = -f_tx_hz * (range_rate / C)
    f_rx_hz = f_tx_hz + doppler_shift_hz

    return {
        "f_rx_hz": float(f_rx_hz),
        "doppler_shift_hz": float(doppler_shift_hz),
    }


def calculate_atmospheric_attenuation(elevation_deg: float, frequency_hz: float) -> float:
    """Calculate atmospheric attenuation using a simplified cosecant model."""
    if frequency_hz < 3e9:
        a_zenith = 0.03
    elif frequency_hz < 10e9:
        a_zenith = 0.05
    elif frequency_hz < 18e9:
        a_zenith = 0.15
    else:
        a_zenith = 0.35

    elevation_rad = np.radians(max(elevation_deg, 5.0))
    return float(a_zenith / np.sin(elevation_rad))
