"""
Time system conversions (UTC, TAI, GPS, TT, TDB) and Julian date utilities.
"""

from __future__ import annotations

import numpy as np


def calc_jd(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    sec: float = 0.0
) -> tuple[float, float]:
    """
    Calculate the Julian Date (JD) from a Gregorian date and time.

    Standard epoch is Noon, January 1, 4713 BC.

    Parameters
    ----------
    year : int
        Year (e.g. 2024).
    month : int
        Month (1-12).
    day : int
        Day of month (1-31).
    hour : int, optional
        Hour (0-23). Default 0.
    minute : int, optional
        Minute (0-59). Default 0.
    sec : float, optional
        Seconds (0-60). Default 0.0.

    Returns
    -------
    tuple[float, float]
        (JD integer part, JD fractional part).
    """
    if month <= 2:
        y_adj = int(year) - 1
        m_adj = int(month) + 12
    else:
        y_adj = int(year)
        m_adj = int(month)

    # 1. Compute integer part (Meeus algorithm)
    a_val = int(y_adj / 100)
    b_val = 2 - a_val + int(a_val / 4)
    jd_int = int(365.25 * (y_adj + 4716)) + int(30.6001 * (m_adj + 1)) + day + b_val - 1524.5

    # 2. Compute fractional day part
    jd_frac = (hour * 3600.0 + minute * 60.0 + sec) / 86400.0

    # Normalize
    rollover = np.floor(jd_frac)
    jd_int += rollover
    jd_frac -= rollover

    return float(jd_int), float(jd_frac)


def jd_to_datetime(jd: float, jd_frac: float) -> tuple[int, int, int, int, int, float]:
    """
    Convert Julian Date to Gregorian date and time.

    Parameters
    ----------
    jd : float
        Integer part of Julian Date.
    jd_frac : float
        Fractional part of Julian Date.

    Returns
    -------
    tuple[int, int, int, int, int, float]
        (Year, Month, Day, Hour, Minute, Second).
    """
    jd_total = float(jd) + float(jd_frac)
    z_val = int(jd_total + 0.5)
    f_val = jd_total + 0.5 - z_val

    if z_val < 2299161:
        a_val = z_val
    else:
        alpha = int((z_val - 1867216.25) / 36524.25)
        a_val = z_val + 1 + alpha - int(alpha / 4)

    b_val = a_val + 1524
    c_val = int((b_val - 122.1) / 365.25)
    d_val = int(365.25 * c_val)
    e_val = int((b_val - d_val) / 30.6001)

    day = b_val - d_val - int(30.6001 * e_val) + f_val
    month = e_val - 1 if e_val < 14 else e_val - 13
    year = c_val - 4716 if month > 2 else c_val - 4715

    # Extract time from fractional day
    day_int = int(day)
    time_frac = (day - day_int) * 86400.0

    hour = int(time_frac / 3600.0)
    minute = int((time_frac - hour * 3600.0) / 60.0)
    second = time_frac - hour * 3600.0 - minute * 60.0

    return int(year), int(month), day_int, hour, minute, second


def day_to_mdtime(year: int, doy_frac: float) -> tuple[int, int, int, int, float]:
    """
    Convert day-of-year (DOY) to month, day, and time.

    Parameters
    ----------
    year : int
        Year.
    doy_frac : float
        Day of the year, including fraction (1.0 = Jan 1 00:00).

    Returns
    -------
    tuple[int, int, int, int, float]
        (Month, Day, Hour, Minute, Second).
    """
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    mon_days = [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    doy = int(doy_frac)
    time_frac = (float(doy_frac) - doy) * 24.0

    month = 0
    while doy > mon_days[month] and month < 11:
        doy -= mon_days[month]
        month += 1

    hour = int(time_frac)
    minute = int((time_frac - hour) * 60.0)
    second = ((time_frac - hour) * 60.0 - minute) * 60.0

    return month + 1, doy, hour, minute, second


def calc_gmst(jd: float, dut1: float = 0.0) -> float:
    """
    Calculates Greenwich Mean Sidereal Time from Julian Date and DUT1.

    Parameters
    ----------
    jd : float
        Julian Date (UT1).
    dut1 : float, optional
        UT1-UTC difference (s). Default is 0.0.

    Returns
    -------
    float
        Greenwich Mean Sidereal Time in radians.
    """
    jd_ut1 = jd + dut1 / 86400.0
    ut1 = (jd_ut1 - 2451545.0) / 36525.0
    gmst = (
        67310.54841 + (876600 * 3600.0 + 8640184.812866) * ut1 + 0.093104 * ut1**2 - 6.2e-6 * ut1**3
    ) % 86400.0
    gmst /= 240.0
    return float(np.radians(gmst % 360.0))


def calc_last(jd: float, lon: float, dut1: float = 0.0) -> float:
    """
    Calculates Local Apparent Sidereal Time from Julian Date and Longitude.

    Parameters
    ----------
    jd : float
        Julian Date (UT1).
    lon : float
        Longitude (rad).
    dut1 : float, optional
        UT1-UTC difference (s). Default is 0.0.

    Returns
    -------
    float
        Local Apparent Sidereal Time in radians.
    """
    ut1 = (jd - 2451545.0) / 36525.0
    gmst = calc_gmst(jd + dut1 / 86400.0)

    eps = np.radians(23.439291 - 0.0130042 * ut1)  # Obliquity of ecliptic
    omega = np.radians(
        125.04452 - 1934.136261 * ut1
    )  # Nutation in Longitude of the ascending node of the Moon's orbit
    l_sun = np.radians(
        280.4665 + 36000.7698 * ut1 + 0.0003032 * ut1**2
    )  # Mean longitude of the Sun
    d_psi = (
        np.radians(-17.20 * np.sin(omega) - 1.32 * np.sin(2 * l_sun) + 0.216 * np.sin(2 * eps))
        / 3600.0
    )  # Nutation in Longitude

    equinox = d_psi * np.cos(eps)
    # GAST = GMST + equinox
    last = gmst + equinox + lon
    last = last % (2 * np.pi)
    return float(last)


def calc_lst(gmst: float, lon: float, dut1: float = 0.0) -> float:
    """
    Calculates Local Sidereal Time from GMST and Longitude.

    Parameters
    ----------
    gmst : float
        Greenwich Mean Sidereal Time (rad).
    lon : float
        Longitude (rad).
    dut1 : float, optional
        UT1-UTC difference (s). Default is 0.0.

    Returns
    -------
    float
        Local Sidereal Time in radians.
    """
    return gmst + lon + dut1 / 86400.0


def calc_doy(year: int, month: int, day: int) -> int:
    """
    Calculates the day of the year.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month (1-12).
    day : int
        Day of the month (1-31).

    Returns
    -------
    int
        Day of the year (1-366).
    """
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap_year(year):
        months[1] = 29
    idx = month - 1
    doy = np.sum(months[:idx]) + day
    return int(doy)


def is_leap_year(year: int) -> bool:
    """
    Checks if a year is a leap year.

    Parameters
    ----------
    year : int
        Year.

    Returns
    -------
    bool
        True if the year is a leap year, False otherwise.
    """
    if np.remainder(year, 4) != 0:
        return False
    else:
        if np.remainder(year, 100) == 0:
            if np.remainder(year, 400) == 0:
                return True
            else:
                return False
        else:
            return True


def convert_time(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    sec: float,
    timezone: float,
    output: str,
    dut1: float,
    dat: float,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """
    Converts UTC to any time system.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    day : int
        Day.
    hour : int
        Hour.
    minute : int
        Minute.
    sec : float
        Seconds.
    timezone : float
        Timezone offset (hours).
    output : str
        Target time system (placeholder).
    dut1 : float
        UT1 - UTC (s).
    dat : float
        TAI - UTC (s).

    Returns
    -------
    tuple
        A tuple containing converted time values for various systems:
        (ut1, tut1, jdut1, jdut1frac, utc, tai, gps, tt, ttt, jdtt, jdttfrac, tdb, ttdb, jdtdb, jdtdbfrac)
    """
    utc = hour * 3600 + minute * 60 + sec

    ut1 = utc + dut1
    hour_temp = int(ut1 / 3600)
    minute_temp = int((ut1 / 3600 - hour_temp) * 60)
    second_temp = ((ut1 / 3600 - hour_temp) - minute_temp / 60) * 3600
    second_temp = (
        round(second_temp) if abs(second_temp - round(second_temp)) < 1e-10 else second_temp
    )
    jdut1, jdut1frac = calc_jd(year, month, day, hour_temp, minute_temp, second_temp)
    tut1 = (jdut1 + jdut1frac - 2451545) / 36525

    tai = utc + dat

    gps = tai - 19

    tt = tai + 32.184
    hour_temp = int(tt / 3600)
    minute_temp = int((tt / 3600 - hour_temp) * 60)
    second_temp = ((tt / 3600 - hour_temp) - minute_temp / 60) * 3600
    second_temp = (
        round(second_temp) if abs(second_temp - round(second_temp)) < 1e-10 else second_temp
    )
    jdtt, jdttfrac = calc_jd(year, month, day, hour_temp, minute_temp, second_temp)
    tutt = (jdtt + jdttfrac - 2451545) / 36525
    ttt = tutt  # define ttt for use in TDB calc

    tdb = (
        tt
        + 0.001657 * np.sin(628.3076 * ttt + 6.2401)
        + 0.000022 * np.sin(575.3385 * ttt + 4.297)
        + 0.000014 * np.sin(1256.6152 * ttt + 6.1969)
        + 0.000005 * np.sin(606.9777 * ttt + 4.0212)
        + 0.000005 * np.sin(52.9691 * ttt + 0.4444)
        + 0.000002 * np.sin(21.3299 * ttt + 5.5431)
        + 0.00001 * ttt * np.sin(628.3076 * ttt + 4.249)
    )
    hour_temp = int(tdb / 3600)
    minute_temp = int((tdb / 3600 - hour_temp) * 60)
    second_temp = ((tdb / 3600 - hour_temp) - minute_temp / 60) * 3600
    second_temp = (
        round(second_temp) if abs(second_temp - round(second_temp)) < 1e-10 else second_temp
    )
    jdtdb, jdtdbfrac = calc_jd(year, month, day, hour_temp, minute_temp, second_temp)
    ttdb = (jdtdb + jdtdbfrac - 2451545) / 36525

    return (
        ut1,
        tut1,
        jdut1,
        jdut1frac,
        utc,
        tai,
        gps,
        tt,
        ttt,
        jdtt,
        jdttfrac,
        tdb,
        ttdb,
        jdtdb,
        jdtdbfrac,
    )




