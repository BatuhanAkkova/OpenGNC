"""Multiplicative Extended Kalman Filter (MEKF) for attitude estimation."""

from __future__ import annotations

import numpy as np

from opengnc.sensors.sensor import SensorMeasurement
from opengnc.utils.quat_utils import (
    quat_conj,
    quat_mult,
    quat_normalize,
    quat_rot,
    skew_symmetric,
)


class MEKF:
    """Packet-oriented multiplicative EKF for spacecraft attitude estimation."""

    VECTOR_QUANTITIES = {"sun_vector", "magnetic_field", "nadir_vector"}

    def __init__(
        self,
        q_init: np.ndarray | None = None,
        beta_init: np.ndarray | None = None,
    ) -> None:
        if q_init is None:
            self.q = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            self.q = quat_normalize(np.asarray(q_init, dtype=float))

        if beta_init is None:
            self.beta = np.zeros(3)
        else:
            self.beta = np.asarray(beta_init, dtype=float)

        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.001
        self.R = np.eye(3) * 0.01
        self.x = np.concatenate([self.q, self.beta])

    def predict(
        self,
        measurement: SensorMeasurement,
        dt: float | None = None,
        q_mat: np.ndarray | None = None,
    ) -> None:
        """Propagate state from an angular-rate measurement packet."""
        if measurement.quantity != "angular_rate":
            raise ValueError("MEKF.predict expects an 'angular_rate' measurement packet.")

        step = dt if dt is not None else measurement.metadata.get("sample_period_s")
        if step is None:
            raise ValueError(
                "A timestep is required either as dt or measurement.metadata['sample_period_s']."
            )

        self._predict_raw(
            np.asarray(measurement.value, dtype=float),
            float(step),
            self.Q if q_mat is None else q_mat,
        )

    def update(
        self,
        measurement: SensorMeasurement,
        r_mat: np.ndarray | None = None,
    ) -> None:
        """Apply a vector or quaternion correction from a measurement packet."""
        if measurement.quantity == "attitude_quaternion":
            self.update_quaternion(measurement, r_mat)
            return

        if measurement.quantity not in self.VECTOR_QUANTITIES:
            raise ValueError(f"Unsupported MEKF measurement quantity: {measurement.quantity}")

        reference = measurement.metadata.get("reference")
        if reference is None:
            raise ValueError(
                f"Measurement '{measurement.quantity}' requires metadata['reference'] for the inertial vector."
            )

        self._update_vector_raw(
            np.asarray(measurement.value, dtype=float),
            np.asarray(reference, dtype=float),
            self.R if r_mat is None else r_mat,
        )

    def update_quaternion(
        self,
        measurement: SensorMeasurement,
        r_mat: np.ndarray | None = None,
    ) -> None:
        """Apply a direct attitude correction from a quaternion measurement packet."""
        if measurement.quantity != "attitude_quaternion":
            raise ValueError("MEKF.update_quaternion expects an 'attitude_quaternion' packet.")

        default_r = np.eye(3) * 0.01
        self._update_quaternion_raw(
            np.asarray(measurement.value, dtype=float),
            default_r if r_mat is None else r_mat,
        )

    def _predict_raw(self, omega_meas: np.ndarray, dt: float, q_mat: np.ndarray) -> None:
        qm = np.asarray(q_mat, dtype=float)
        w_meas = np.asarray(omega_meas, dtype=float)

        omega = w_meas - self.beta
        wm = np.linalg.norm(omega)

        if wm > 1e-10:
            axis = omega / wm
            angle = wm * dt
            dq = np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])
            self.q = quat_mult(self.q, dq)

        self.q = quat_normalize(self.q)

        wx = skew_symmetric(omega)
        f_jac = np.zeros((6, 6))
        f_jac[0:3, 0:3] = -wx
        f_jac[0:3, 3:6] = -np.eye(3)

        phi = np.eye(6) + f_jac * dt
        self.P = (phi @ self.P @ phi.T) + (qm * dt)
        self.x = np.concatenate([self.q, self.beta])

    def _update_vector_raw(self, z_body: np.ndarray, z_ref: np.ndarray, r_mat: np.ndarray) -> None:
        r = np.asarray(r_mat, dtype=float)
        zb = np.asarray(z_body, dtype=float)
        zr = np.asarray(z_ref, dtype=float)

        zb /= np.linalg.norm(zb)
        zr /= np.linalg.norm(zr)

        q_inv = quat_conj(self.q)
        zp = quat_rot(q_inv, zr)

        h_mat = np.zeros((3, 6))
        h_mat[:, 0:3] = skew_symmetric(zp)

        s_mat = (h_mat @ self.P @ h_mat.T) + r
        k_gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)

        dx = k_gain @ (zb - zp)
        dtheta = dx[0:3]
        dbeta = dx[3:6]

        dq_corr = np.concatenate([0.5 * dtheta, [1.0]])
        self.q = quat_normalize(quat_mult(self.q, dq_corr))
        self.beta += dbeta

        i_kh = np.eye(6) - (k_gain @ h_mat)
        self.P = (i_kh @ self.P @ i_kh.T) + (k_gain @ r @ k_gain.T)
        self.x = np.concatenate([self.q, self.beta])

    def _update_quaternion_raw(self, q_meas: np.ndarray, r_mat: np.ndarray) -> None:
        r = np.asarray(r_mat, dtype=float)
        q_obs = quat_normalize(np.asarray(q_meas, dtype=float))
        dq = quat_mult(quat_conj(self.q), q_obs)
        if dq[3] < 0:
            dq = -dq
        innovation = 2.0 * dq[:3]

        h_mat = np.zeros((3, 6))
        h_mat[:, 0:3] = np.eye(3)

        s_mat = (h_mat @ self.P @ h_mat.T) + r
        k_gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)

        dx = k_gain @ innovation
        dtheta = dx[0:3]
        dbeta = dx[3:6]

        dq_corr = np.concatenate([0.5 * dtheta, [1.0]])
        self.q = quat_normalize(quat_mult(self.q, dq_corr))
        self.beta += dbeta

        i_kh = np.eye(6) - (k_gain @ h_mat)
        self.P = (i_kh @ self.P @ i_kh.T) + (k_gain @ r @ k_gain.T)
        self.x = np.concatenate([self.q, self.beta])
