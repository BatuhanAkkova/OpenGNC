"""
Generic PID controller implementation with anti-windup logic.
"""

from __future__ import annotations


class PID:
    r"""
    Generic PID controller with anti-windup.

    Control Law:
    $u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$

    Parameters
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    output_limits : tuple[float, float] | None, optional
        (min, max) saturation limits.
    anti_windup_method : str, optional
        Method (e.g., "clamping"). Default "clamping".
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: tuple[float, float] | None = None,
        anti_windup_method: str = "clamping",
    ) -> None:
        """Initialize the PID controller instance."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.anti_windup_method = anti_windup_method

        self.integral_error = 0.0
        self.previous_error = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset the internal integrator and error states to zero."""
        self.integral_error = 0.0
        self.previous_error = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        Update the PID control calculation for a single time step.

        Parameters
        ----------
        error : float
            The current error signal (setpoint - measured).
        dt : float
            Time step since the last update (s).

        Returns
        -------
        float
            The computed control output signal.
        """
        if dt <= 0:
            return 0.0

        # Proportional term
        p_term = self.kp * error

        # Integral term calculation (tentative)
        self.integral_error += error * dt
        i_term = self.ki * self.integral_error

        # Derivative term using backward difference
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        # Calculate raw output
        output = p_term + i_term + d_term

        # Apply output limits and anti-windup
        if self.output_limits is not None:
            min_limit, max_limit = self.output_limits

            # Saturation check
            if output > max_limit:
                output_clamped = max_limit
            elif output < min_limit:
                output_clamped = min_limit
            else:
                output_clamped = output

            # Anti-windup (Clamping method)
            # If the output is saturated AND the error has the same sign as the
            # control signal (driving it further into saturation), stop integrating.
            if self.anti_windup_method == "clamping":
                if output != output_clamped:
                    if (output > max_limit and error > 0) or (output < min_limit and error < 0):
                        # Revert the integration step
                        self.integral_error -= error * dt

            output = output_clamped

        self.previous_error = error
        return output
