"""
Gravitational acceleration models (Two-Body, J2, Harmonics) and Gradient Torques.
"""

import csv
import os
from typing import cast

import numpy as np
from numba import njit

from opengnc.utils.frame_conversion import ecef2eci, eci2ecef
from opengnc.utils.quat_utils import quat_conj, quat_rot


class TwoBodyGravity:
    r"""
    Standard Point-Mass Gravity.

    Acceleration:
    $\mathbf{a}_{gg} = -\frac{\mu}{r^3} \mathbf{r}$

    Parameters
    ----------
    mu : float, optional
        Gravitational parameter ($m^3/s^2$). Default Earth.
    """

    def __init__(self, mu: float = 398600.4418e9) -> None:
        """Initialize with gravitational constant."""
        self.mu = mu

    def get_acceleration(self, r_eci: np.ndarray, jd: float | None = None) -> np.ndarray:
        """
        Calculate point mass acceleration.

        Parameters
        ----------
        r_eci : np.ndarray
            ECI Position (m).
        jd : float | None, optional
            Julian Date.

        Returns
        -------
        np.ndarray
            Acceleration ($m/s^2$).
        """
        r_vec = np.asarray(r_eci, dtype=float)
        r_mag = np.linalg.norm(r_vec)
        return cast(np.ndarray, -self.mu / r_mag**3 * r_vec)


class J2Gravity:
    r"""
    Oblateness Perturbation ($J_2$).

    Specific Acceleration:
    $a_{J2} = \frac{3\mu J_2 R_e^2}{2r^5} \left[ (5\frac{z^2}{r^2}-1)x, (5\frac{z^2}{r^2}-1)y, (5\frac{z^2}{r^2}-3)z \right]$

    Parameters
    ----------
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).
    j2 : float, optional
        J2 coefficient.
    re : float, optional
        Equatorial radius (m).
    """

    def __init__(self, mu: float = 398600.4418e9, j2: float = 0.001082635855, re: float = 6378137.0) -> None:
        """Initialize J2 parameters."""
        self.mu = mu
        self.j2 = j2
        self.re = re

    def get_acceleration(self, r_eci: np.ndarray, jd: float | None = None) -> np.ndarray:
        r"""
        Calculate acceleration including $J_2$ perturbation.

        Formula:
        $a_{j2,x} = \frac{3}{2} J_2 \frac{\mu}{r^2} \frac{R_e^2}{r^2} \frac{x}{r} (5\frac{z^2}{r^2} - 1)$

        Parameters
        ----------
        r_eci : np.ndarray
            ECI Position (m).
        jd : float | None, optional
            Julian Date.

        Returns
        -------
        np.ndarray
            Acceleration vector ($m/s^2$).
        """
        r_vec = np.asarray(r_eci, dtype=float)
        r_mag = np.linalg.norm(r_vec)
        x, y, z = r_vec

        factor = (1.5 * self.j2 * self.mu * self.re**2) / (r_mag**5)

        ax = factor * x * (5 * (z / r_mag)**2 - 1)
        ay = factor * y * (5 * (z / r_mag)**2 - 1)
        az = factor * z * (5 * (z / r_mag)**2 - 3)

        return cast(np.ndarray, np.array([ax, ay, az]) + TwoBodyGravity(self.mu).get_acceleration(r_vec))


class HarmonicsGravity:
    """
    High-Fidelity Spherical Harmonics Model (EGM2008).

    Calculates the fine-grained gravitational acceleration by expanding the
    potential as an infinite series of Legendre polynomials and associated
    functions.

    Parameters
    ----------
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).
    re : float, optional
        Planetary reference radius (m).
    n_max : int, optional
        Maximum expansion degree.
    m_max : int, optional
        Maximum expansion order.
    file_path : Optional[str], optional
        Path to harmonic coefficients CSV.
    """

    def __init__(
        self,
        mu: float = 398600.4418e9,
        re: float = 6378137.0,
        n_max: int = 20,
        m_max: int = 20,
        file_path: str | None = None
    ) -> None:
        """Load potential coefficients and initialize recursion workspace."""
        self.mu = mu
        self.re = re
        self.n_max = n_max
        self.m_max = m_max

        self.C = np.zeros((n_max + 1, m_max + 1))
        self.S = np.zeros((n_max + 1, m_max + 1))

        if file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base_dir, "egm2008.csv")

        self._load_coefficients(file_path)

    def _load_coefficients(self, file_path: str) -> None:
        """Internal parser for harmonic datasets."""
        if not os.path.exists(file_path):
            print("Warning: Gravity coefficient file not found")
            return

        with open(file_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                n, m = int(row[0]), int(row[1])
                cv, sv = float(row[2]), float(row[3])
                if n <= self.n_max and m <= self.m_max:
                    self.C[n, m] = cv
                    self.S[n, m] = sv

    def get_acceleration(self, r_eci: np.ndarray, jd: float) -> np.ndarray:
        """
        Calculate harmonic acceleration vector in ECI.

        Implements Pines-style or Cunningham recursion for high-order gravity.
        Handles coordinate transformation between ECI/ECEF.

        Parameters
        ----------
        r_eci : np.ndarray
            ECI Position (m).
        jd : float
            Julian Date for frame rotation.

        Returns
        -------
        np.ndarray
            Total acceleration vector ($m/s^2$).
        """
        r_val = np.asarray(r_eci)
        r_ecef, _ = eci2ecef(r_val, np.zeros(3), jd)
        x, y, z = r_ecef

        acc_ecef = _compute_gravity_njit(
            x, y, z, self.re, self.mu, self.n_max, self.m_max, self.C, self.S
        )

        aeci, _ = ecef2eci(acc_ecef, np.zeros(3), jd)
        return cast(np.ndarray, aeci)


@njit
def _compute_gravity_njit(
    x: float,
    y: float,
    z: float,
    re: float,
    mu: float,
    n_max: int,
    m_max: int,
    C: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated spherical harmonics recursion.
    """
    r_sq = x*x + y*y + z*z
    r_mag = np.sqrt(r_sq)

    rho = re / r_mag
    sin_phi = z / r_mag
    cos_phi_approx = np.sqrt(max(0.0, 1.0 - sin_phi**2))

    P = np.zeros((n_max + 2, m_max + 2))
    P[0, 0] = 1.0
    # P1,0 = sqrt(3) * sin_phi
    P[1, 0] = np.sqrt(3.0) * sin_phi
    # P1,1 = sqrt(3) * cos_phi
    P[1, 1] = np.sqrt(3.0) * cos_phi_approx

    for n in range(2, n_max + 1):
        # m = 0
        an0 = np.sqrt((4.0 * n**2 - 1.0) / n**2)
        bn0 = np.sqrt((2.0 * n + 1.0) * (n - 1.0)**2 / (n**2 * (2.0 * n - 3.0)))
        P[n, 0] = an0 * sin_phi * P[n-1, 0] - bn0 * P[n-2, 0]

        for m in range(1, min(n + 1, m_max + 1)):
            if n == m:
                cnn = np.sqrt((2.0 * n + 1.0) / (2.0 * n))
                P[n, n] = cnn * cos_phi_approx * P[n-1, n-1]
            else:
                anm = np.sqrt((4.0 * n**2 - 1.0) / (n**2 - m**2))
                bnm = np.sqrt((2.0 * n + 1.0) * ((n - 1.0)**2 - m**2) / ((2.0 * n - 3.0) * (n**2 - m**2)))
                P[n, m] = anm * sin_phi * P[n-1, m] - bnm * P[n-2, m]

    du_dr = -mu / r_sq
    du_dlat, du_dlon = 0.0, 0.0
    lon = np.arctan2(y, x)
    clon, slon = np.cos(lon), np.sin(lon)

    # Precompute m*lon trig
    cos_mlon = np.zeros(m_max + 1)
    sin_mlon = np.zeros(m_max + 1)
    for m in range(m_max + 1):
        cos_mlon[m] = np.cos(m * lon)
        sin_mlon[m] = np.sin(m * lon)

    for n in range(2, n_max + 1):
        rn = rho**n
        s_r, s_lat, s_lon = 0.0, 0.0, 0.0
        for m in range(0, min(n, m_max) + 1):
            cv, sv = C[n, m], S[n, m]
            gt = cv * cos_mlon[m] + sv * sin_mlon[m]

            # Deriv dP/dphi
            if n == m:
                cos_phi_sq = max(1e-15, 1.0 - sin_phi**2)
                dp = -n * sin_phi / np.sqrt(cos_phi_sq) * P[n, n]
            else:
                anm = np.sqrt((2.0 * n + 1.0) / (n - m)) * np.sqrt((2.0 * n - 1.0) / (n + m))
                cos_phi_sq = max(1e-15, 1.0 - sin_phi**2)
                dp = (n * sin_phi * P[n, m] - anm * P[n-1, m]) / np.sqrt(cos_phi_sq)

            s_r += (n + 1.0) * P[n, m] * gt
            s_lat += dp * gt
            s_lon += P[n, m] * m * (-cv * sin_mlon[m] + sv * cos_mlon[m])

        du_dr -= (mu / r_sq) * rn * s_r
        du_dlat += (mu / r_mag) * rn * s_lat
        du_dlon += (mu / r_mag) * rn * s_lon

    safe_cos = max(1e-15, cos_phi_approx)

    term_r = du_dr * cos_phi_approx
    term_lat = (du_dlat / r_mag) * (-sin_phi)
    term_lon = (du_dlon / (r_mag * safe_cos))

    ax_ecef = (term_r + term_lat) * clon + term_lon * (-slon)
    ay_ecef = (term_r + term_lat) * slon + term_lon * clon
    az_ecef = du_dr * sin_phi + (du_dlat / r_mag) * cos_phi_approx

    return np.array([ax_ecef, ay_ecef, az_ecef])


class GradientTorque:
    """
    Gravity Gradient Pointing Restoration Torque.

    Models the restorative moment acting on an asymmetric rigid body
    within a non-uniform gravity field.

    Parameters
    ----------
    mu : float, optional
        Gravitational parameter ($m^3/s^2$).
    """

    def __init__(self, mu: float = 398600.4418e9) -> None:
        """Initialize torque solver."""
        self.mu = mu

    def gravity_gradient_torque(self, J: np.ndarray, r_eci: np.ndarray, q_body2eci: np.ndarray) -> np.ndarray:
        r"""
        Calculate Gravity Gradient torque in Body frame.

        Equation:
        $\mathbf{T}_{gg} = \frac{3\mu}{r^3} \mathbf{u}_n \times (\mathbf{J} \mathbf{u}_n)$

        Parameters
        ----------
        J : np.ndarray
            Inertia tensor ($3 \times 3$) ($kg \cdot m^2$).
        r_eci : np.ndarray
            ECI Position (m).
        q_body2eci : np.ndarray
            Body-to-ECI quaternion $[q_w, q_x, q_y, q_z]$.

        Returns
        -------
        np.ndarray
            Reaction torque vector (3,) (Nm).
        """
        r_val = np.asarray(r_eci)
        rm = np.linalg.norm(r_val)
        if rm < 1e-3:
            return cast(np.ndarray, np.zeros(3))

        u_nadir = -r_val / rm
        q_e2b = quat_conj(np.asarray(q_body2eci))
        u_body = quat_rot(q_e2b, u_nadir)

        return cast(np.ndarray, (3 * self.mu / rm**3) * np.cross(u_body, J @ u_body))


class ThirdBodyGravity:
    """
    Solar and Lunar Gravitational Perturbation Model.

    Calculates the point-mass acceleration acting on the spacecraft due
    to the Sun and Moon.

    Parameters
    ----------
    mu_sun : float, optional
        Solar gravitational parameter.
    mu_moon : float, optional
        Lunar gravitational parameter.
    """

    def __init__(self, mu_sun: float = 1.32712440018e20, mu_moon: float = 4902.800066e9) -> None:
        """Initialize planetary ephemeris models."""
        self.mu_sun = mu_sun
        self.mu_moon = mu_moon
        from opengnc.environment.moon import Moon
        from opengnc.environment.solar import Sun
        self.sun_model = Sun()
        self.moon_model = Moon()

    def get_acceleration(self, r_eci: np.ndarray, jd: float) -> np.ndarray:
        """
        Calculate combined third-body acceleration.

        Parameters
        ----------
        r_eci : np.ndarray
            Satellite ECI position (m).
        jd : float
            Julian Date.

        Returns
        -------
        np.ndarray
            Acceleration vector ($m/s^2$).
        """
        rv = np.asarray(r_eci)
        rss = self.sun_model.calculate_sun_eci(jd)
        rms = self.moon_model.calculate_moon_eci(jd)

        def body_acc(s_vec: np.ndarray, mu: float) -> np.ndarray:
            d = s_vec - rv
            return np.asarray(mu * (d / np.linalg.norm(d) ** 3 - s_vec / np.linalg.norm(s_vec) ** 3))

        return cast(np.ndarray, body_acc(rss, self.mu_sun) + body_acc(rms, self.mu_moon))


class RelativisticCorrection:
    """
    General Relativistic Gravitational Correction.

    Includes static Schwarzschild and dynamic Lense-Thirring
    (frame-dragging) effects.

    Parameters
    ----------
    mu : float, optional
        Gravitational parameter.
    J_earth : Optional[np.ndarray], optional
        Angular momentum vector of the planet.
    """

    def __init__(self, mu: float = 398600.4418e9, J_earth: np.ndarray | None = None) -> None:
        """Initialize with physical constants."""
        self.mu = mu
        self.c = 299792458.0
        if J_earth is None:
            # S vector for Earth
            self.S_vec = np.array([0, 0, 0.3308 * 6378137.0**2 * 7.292115e-5])
        else:
            self.S_vec = np.asarray(J_earth)

    def get_acceleration(self, r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
        """
        Calculate relativistic acceleration correction.

        Parameters
        ----------
        r_eci, v_eci : np.ndarray
            State vectors in ECI.

        Returns
        -------
        np.ndarray
            Correction vector ($m/s^2$).
        """
        r, v = np.asarray(r_eci), np.asarray(v_eci)
        rm, vm = np.linalg.norm(r), np.linalg.norm(v)

        # Schwarzschild static correction
        sch = (self.mu / (self.c**2 * rm**3)) * ( (4*self.mu/rm - vm**2)*r + 4*np.dot(r,v)*v )

        # Lense-Thirring dynamic correction
        lt_term = (2 * self.mu / (self.c**2 * rm**3)) * ( (3.0/rm**2) * np.dot(r, self.S_vec) * r - self.S_vec )

        return cast(np.ndarray, sch + np.cross(lt_term, v))


class OceanTidesGravity:
    """
    Spherical Harmonic Ocean Tide Model (Simplified).

    Applies periodic corrections to the potential field due to
    displacement of oceanic mass.

    Parameters
    ----------
    mu : float, optional
        Gravitational parameter.
    re : float, optional
        Reference radius.
    """

    def __init__(self, mu: float = 398600.4418e9, re: float = 6378137.0) -> None:
        """Initialize with IERS constituent coefficients."""
        self.mu = mu
        self.re = re
        self.coefs = {
            "M2": {"n": 2, "m": 2, "C+": 3.090e-11, "S+": -1.155e-11},
            "S2": {"n": 2, "m": 2, "C+": 0.573e-11, "S+": -0.134e-11},
            "K1": {"n": 2, "m": 1, "C+": -0.155e-11, "S+": 0.621e-11},
            "O1": {"n": 2, "m": 1, "C+": -0.177e-11, "S+": 0.055e-11},
        }

    def _get_doodson_arguments(self, jd: float) -> dict[str, float]:
        """Compute Doodson harmonic arguments for the given epoch."""
        from opengnc.utils.time_utils import calc_gmst
        T = (jd - 2451545.0) / 36525.0
        n = jd - 2451545.0
        gmst = calc_gmst(jd)
        s = np.radians((218.316 + 481267.8813 * T) % 360)
        h = np.radians((280.459 + 0.98564736 * n) % 360)

        return {
            "M2": 2 * (gmst + np.pi - s),
            "S2": 2 * (gmst + np.pi - h),
            "K1": gmst + np.pi,
            "O1": gmst + np.pi - 2 * s,
        }

    def get_acceleration(self, r_eci: np.ndarray, jd: float) -> np.ndarray:
        """
        Calculate tidal acceleration contribution in ECI.

        Parameters
        ----------
        r_eci : np.ndarray
            ECI Position.
        jd : float
            Julian Date.

        Returns
        -------
        np.ndarray
            Correction vector ($m/s^2$).
        """
        rv = np.asarray(r_eci)
        args = self._get_doodson_arguments(jd)

        dc, ds = np.zeros((3, 3)), np.zeros((3, 3))
        for c, v in self.coefs.items():
            n, m = int(v["n"]), int(v["m"])
            a = args[c]
            dc[n, m] += v["C+"] * np.cos(a) + v["S+"] * np.sin(a)
            ds[n, m] += v["S+"] * np.cos(a) - v["C+"] * np.sin(a)

        r_ecef, _ = eci2ecef(rv, np.zeros(3), jd)
        x, y, z = r_ecef
        rm = np.linalg.norm(r_ecef)
        lat = np.arcsin(z / rm)
        lon = np.arctan2(y, x)

        rho = (self.re / rm)**2
        p_21 = np.sqrt(15) * np.sin(lat) * np.cos(lat)
        p_22 = np.sqrt(15/4) * np.cos(lat)**2

        u_21 = rho * p_21 * (dc[2, 1] * np.cos(lon) + ds[2, 1] * np.sin(lon))
        u_22 = rho * p_22 * (dc[2, 2] * np.cos(2*lon) + ds[2, 2] * np.sin(2*lon))

        acc_ecef = - (self.mu / rm**2) * (u_21 + u_22) * (r_ecef / rm)
        aeci, _ = ecef2eci(acc_ecef, np.zeros(3), jd)
        return cast(np.ndarray, aeci)




