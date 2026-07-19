"""Unscented Kalman Filter (UKF) with support for states on manifolds."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.linalg import cholesky, sqrtm

from opengnc.sensors.sensor import SensorMeasurement
from opengnc.utils.quat_utils import (
    axis_angle_to_quat,
    quat_conj,
    quat_mult,
    quat_normalize,
    quat_rot,
)


class UKF:
    r"""Generalized Unscented Kalman Filter (UKF)."""

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dim_p: int | None = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        subtract_x: Callable[..., np.ndarray] | None = None,
        add_x: Callable[..., np.ndarray] | None = None,
        mean_x: Callable[..., np.ndarray] | None = None,
    ) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_p = dim_p if dim_p is not None else dim_x

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = alpha**2 * (self.dim_p + kappa) - self.dim_p
        self.gamma = np.sqrt(self.dim_p + self.lambda_)
        self.num_sigmas = 2 * self.dim_p + 1

        self.Wm = np.zeros(self.num_sigmas)
        self.Wc = np.zeros(self.num_sigmas)
        self.Wm[0] = self.lambda_ / (self.dim_p + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_p + self.lambda_) + (1 - alpha**2 + beta)

        w = 1.0 / (2 * (self.dim_p + self.lambda_))
        for i in range(1, self.num_sigmas):
            self.Wm[i] = w
            self.Wc[i] = w

        self.subtract_x = subtract_x if subtract_x is not None else lambda x1, x2: x1 - x2
        self.add_x = add_x if add_x is not None else lambda x, dx: x + dx
        self.mean_x = (
            mean_x if mean_x is not None else lambda sigmas, weights: np.dot(weights, sigmas)
        )

        self.x = np.zeros(dim_x)
        self.P = np.eye(self.dim_p)
        self.Q = np.eye(self.dim_p)
        self.R = np.eye(dim_z)

    def predict(
        self,
        dt: float,
        fx_func: Callable[..., np.ndarray],
        q_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        q = np.asarray(q_mat) if q_mat is not None else self.Q

        sigmas = self.generate_sigma_points(self.x, self.P)
        sigmas_f = []
        for i in range(self.num_sigmas):
            sigmas_f.append(fx_func(sigmas[i], dt, **kwargs))
        sigmas_f_arr = np.array(sigmas_f)

        self.x = self.mean_x(sigmas_f_arr, self.Wm)

        self.P = np.zeros((self.dim_p, self.dim_p))
        for i in range(self.num_sigmas):
            dx = self.subtract_x(sigmas_f_arr[i], self.x)
            self.P += self.Wc[i] * np.outer(dx, dx)
        self.P += q * dt

    def update(
        self,
        z: np.ndarray,
        hx_func: Callable,
        r_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        r = np.asarray(r_mat) if r_mat is not None else self.R
        zv = np.asarray(z)

        sigmas_f = self.generate_sigma_points(self.x, self.P)
        sigmas_h = []
        for i in range(self.num_sigmas):
            sigmas_h.append(hx_func(sigmas_f[i], **kwargs))
        sigmas_h_arr = np.array(sigmas_h)

        zp = np.dot(self.Wm, sigmas_h_arr)

        s_mat = np.zeros((self.dim_z, self.dim_z))
        pxz = np.zeros((self.dim_p, self.dim_z))

        for i in range(self.num_sigmas):
            dz = sigmas_h_arr[i] - zp
            dx = self.subtract_x(sigmas_f[i], self.x)
            s_mat += self.Wc[i] * np.outer(dz, dz)
            pxz += self.Wc[i] * np.outer(dx, dz)

        s_mat += r

        k_gain = pxz @ np.linalg.inv(s_mat)
        self.x = self.add_x(self.x, k_gain @ (zv - zp))
        self.P = self.P - (k_gain @ s_mat @ k_gain.T)

    def generate_sigma_points(self, x: np.ndarray, p_cov: np.ndarray) -> np.ndarray:
        sigmas = [x]
        p_sym = (p_cov + p_cov.T) / 2 + np.eye(self.dim_p) * 1e-12

        try:
            l_mat = cholesky((self.dim_p + self.lambda_) * p_sym, lower=True)
            for i in range(self.dim_p):
                sigmas.append(self.add_x(x, l_mat[:, i]))
                sigmas.append(self.add_x(x, -l_mat[:, i]))
        except np.linalg.LinAlgError:
            u_mat = sqrtm((self.dim_p + self.lambda_) * p_sym).real
            for i in range(self.dim_p):
                sigmas.append(self.add_x(x, u_mat[i]))
                sigmas.append(self.add_x(x, -u_mat[i]))

        return np.array(sigmas)


class UKF_Attitude(UKF):
    """Packet-oriented UKF specialized for spacecraft attitude estimation."""

    VECTOR_QUANTITIES = {"sun_vector", "magnetic_field", "nadir_vector"}

    def __init__(
        self,
        q_init: np.ndarray | None = None,
        bias_init: np.ndarray | None = None,
        dim_z: int = 3,
        **kwargs: Any,
    ) -> None:
        self._quat_mult = quat_mult
        self._quat_conj = quat_conj
        self._axis_angle_to_quat = axis_angle_to_quat
        self._quat_normalize = quat_normalize

        def subtract_x(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            dq = self._quat_mult(self._quat_conj(x2[:4]), x1[:4])
            if dq[3] < 0:
                dq *= -1
            dtheta = 2 * dq[:3]
            dbias = x1[4:] - x2[4:]
            return np.concatenate([dtheta, dbias])

        def add_x(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
            dq = self._axis_angle_to_quat(dx[:3])
            q_new = self._quat_normalize(self._quat_mult(x[:4], dq))
            bias_new = x[4:] + dx[3:]
            return np.concatenate([q_new, bias_new])

        def mean_x(sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
            q_ref = sigmas[0, :4]
            q_avg = np.zeros(4)
            for i in range(len(weights)):
                q = sigmas[i, :4]
                if np.dot(q, q_ref) < 0:
                    q = -q
                q_avg += weights[i] * q

            q_avg = self._quat_normalize(q_avg)
            bias_avg = np.dot(weights, sigmas[:, 4:])
            return np.concatenate([q_avg, bias_avg])

        if "alpha" not in kwargs:
            kwargs["alpha"] = 1e-2

        super().__init__(
            dim_x=7,
            dim_z=dim_z,
            dim_p=6,
            subtract_x=subtract_x,
            add_x=add_x,
            mean_x=mean_x,
            **kwargs,
        )

        if q_init is None:
            q_init = np.array([0.0, 0.0, 0.0, 1.0])
        if bias_init is None:
            bias_init = np.zeros(3)
        self.x = np.concatenate([np.asarray(q_init, dtype=float), np.asarray(bias_init, dtype=float)])

    def predict(  # type: ignore[override]
        self,
        measurement: SensorMeasurement,
        dt: float | None = None,
        q_mat: np.ndarray | None = None,
    ) -> None:
        """Propagate the attitude state from an angular-rate measurement packet."""
        if measurement.quantity != "angular_rate":
            raise ValueError("UKF_Attitude.predict expects an 'angular_rate' measurement packet.")

        step = dt if dt is not None else measurement.metadata.get("sample_period_s")
        if step is None:
            raise ValueError(
                "A timestep is required either as dt or measurement.metadata['sample_period_s']."
            )

        super().predict(
            float(step),
            self._process_model,
            self.Q if q_mat is None else q_mat,
            omega_meas=np.asarray(measurement.value, dtype=float),
        )

    def update(  # type: ignore[override]
        self,
        measurement: SensorMeasurement,
        r_mat: np.ndarray | None = None,
    ) -> None:
        """Apply a vector or quaternion correction from a measurement packet."""
        if measurement.quantity == "attitude_quaternion":
            self._update_quaternion(measurement, r_mat)
            return

        if measurement.quantity not in self.VECTOR_QUANTITIES:
            raise ValueError(f"Unsupported UKF attitude measurement quantity: {measurement.quantity}")

        reference = measurement.metadata.get("reference")
        if reference is None:
            raise ValueError(
                f"Measurement '{measurement.quantity}' requires metadata['reference'] for the inertial vector."
            )

        body_vec = np.asarray(measurement.value, dtype=float)
        body_vec /= np.linalg.norm(body_vec)
        ref_vec = np.asarray(reference, dtype=float)
        ref_vec /= np.linalg.norm(ref_vec)

        super().update(
            body_vec,
            self._vector_measurement_model,
            self.R if r_mat is None else r_mat,
            z_ref=ref_vec,
        )

    def _process_model(self, x: np.ndarray, dt: float, omega_meas: np.ndarray) -> np.ndarray:
        q = x[:4]
        bias = x[4:7]
        omega_body = np.asarray(omega_meas, dtype=float) - bias
        omega_norm = np.linalg.norm(omega_body)
        if omega_norm > 1e-12:
            axis = omega_body / omega_norm
            angle = omega_norm * dt
            dq = np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])
            q_new = quat_normalize(quat_mult(q, dq))
        else:
            q_new = q
        return np.concatenate([q_new, bias])

    @staticmethod
    def _vector_measurement_model(x: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
        return quat_rot(quat_conj(x[:4]), z_ref)

    @staticmethod
    def _quaternion_measurement_model(x: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
        dq = quat_mult(quat_conj(q_reference), x[:4])
        if dq[3] < 0:
            dq = -dq
        return 2.0 * dq[:3]

    def _update_quaternion(
        self,
        measurement: SensorMeasurement,
        r_mat: np.ndarray | None = None,
    ) -> None:
        super().update(
            np.zeros(3),
            self._quaternion_measurement_model,
            self.R if r_mat is None else r_mat,
            q_reference=quat_normalize(np.asarray(measurement.value, dtype=float)),
        )
