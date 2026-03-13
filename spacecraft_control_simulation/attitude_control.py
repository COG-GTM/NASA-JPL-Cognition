"""
Spacecraft Attitude Control Simulation

Full 3-DOF spacecraft attitude dynamics and control simulation using
quaternion kinematics and PID + reaction wheel control — the Python
equivalent of a Simulink attitude control model.

Inspired by:
- nasa-jpl/SAAS: System-level autonomy simulation
- JPL's MONTE tool: Mission analysis and operations

This replaces what typically requires a Simulink model with a pure Python
simulation that can be version-controlled, tested, and extended by AI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class SpacecraftConfig:
    """Spacecraft physical configuration for attitude simulation."""

    # Moments of inertia (kg*m^2) — typical small satellite
    inertia_matrix: np.ndarray = field(default_factory=lambda: np.diag([50.0, 60.0, 40.0]))
    # Reaction wheel properties
    rw_max_torque_nm: float = 0.05  # Max torque per wheel (Nm)
    rw_max_momentum_nms: float = 1.0  # Max angular momentum (Nms)
    rw_spin_axis: np.ndarray = field(
        default_factory=lambda: np.eye(3)  # 3 orthogonal wheels
    )
    # Disturbance torques
    gravity_gradient_enabled: bool = True
    solar_pressure_enabled: bool = True
    magnetic_torque_enabled: bool = True
    # Orbit parameters for disturbance calculation
    orbital_altitude_km: float = 500.0
    orbital_period_s: float = 5670.0  # ~94.5 min LEO


@dataclass
class PIDGains:
    """PID controller gains for attitude control."""

    kp: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.01]))
    kd: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0]))


@dataclass
class SimulationState:
    """Complete state of the attitude simulation at a time step."""

    time: float
    quaternion: np.ndarray  # [q1, q2, q3, q0] (scalar last)
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    control_torque: np.ndarray  # [Tx, Ty, Tz] Nm
    pointing_error_deg: float
    rw_momentum: np.ndarray  # Reaction wheel stored momentum


@dataclass
class SimulationResult:
    """Complete simulation history."""

    time: np.ndarray
    quaternions: np.ndarray  # (N, 4)
    angular_velocities: np.ndarray  # (N, 3)
    control_torques: np.ndarray  # (N, 3)
    pointing_errors: np.ndarray  # (N,)
    euler_angles: np.ndarray  # (N, 3) in degrees
    disturbance_torques: np.ndarray  # (N, 3)
    settling_time: Optional[float] = None
    max_overshoot_deg: Optional[float] = None
    steady_state_error_deg: Optional[float] = None


class SpacecraftAttitudeController:
    """
    Full spacecraft attitude dynamics and control simulation.

    Simulates 3-DOF rotational dynamics with:
    - Quaternion-based kinematics (singularity-free)
    - PID attitude controller with anti-windup
    - Reaction wheel actuator model with saturation
    - Environmental disturbance torques (gravity gradient, solar pressure, magnetic)
    - Performance metric computation

    This is the Python equivalent of a Simulink attitude control model,
    fully testable and version-controllable.

    Example:
        >>> controller = SpacecraftAttitudeController()
        >>> result = controller.run_simulation(
        ...     initial_attitude_deg=[15, -10, 20],
        ...     target_attitude_deg=[0, 0, 0],
        ...     duration_s=600,
        ... )
        >>> print(f"Settling time: {result.settling_time:.1f} s")
        >>> controller.plot_results(result, output_path="attitude_sim.png")
    """

    def __init__(
        self,
        config: Optional[SpacecraftConfig] = None,
        gains: Optional[PIDGains] = None,
    ):
        self.config = config or SpacecraftConfig()
        self.gains = gains or PIDGains()
        self._integral_error = np.zeros(3)

    def run_simulation(
        self,
        initial_attitude_deg: list[float] | np.ndarray = (15.0, -10.0, 20.0),
        target_attitude_deg: list[float] | np.ndarray = (0.0, 0.0, 0.0),
        initial_rates_dps: list[float] | np.ndarray = (0.5, -0.3, 0.2),
        duration_s: float = 600.0,
        dt: float = 0.1,
    ) -> SimulationResult:
        """
        Run a complete attitude control simulation.

        Args:
            initial_attitude_deg: Initial Euler angles [roll, pitch, yaw] in degrees
            target_attitude_deg: Target Euler angles in degrees
            initial_rates_dps: Initial angular rates in deg/s
            duration_s: Simulation duration in seconds
            dt: Time step in seconds

        Returns:
            SimulationResult with complete history
        """
        initial_attitude_rad = np.deg2rad(initial_attitude_deg)
        target_attitude_rad = np.deg2rad(target_attitude_deg)
        initial_rates_rad = np.deg2rad(initial_rates_dps)

        # Convert Euler angles to quaternions
        q0 = Rotation.from_euler("xyz", initial_attitude_rad).as_quat()  # [x,y,z,w]
        q_target = Rotation.from_euler("xyz", target_attitude_rad).as_quat()

        # Initial state vector: [q0, q1, q2, q3, wx, wy, wz]
        state0 = np.concatenate([q0, initial_rates_rad])

        # Time array
        t_span = (0.0, duration_s)
        t_eval = np.arange(0.0, duration_s, dt)

        # Reset integral error
        self._integral_error = np.zeros(3)

        # Storage for control and disturbance torques
        control_history: list[np.ndarray] = []
        disturbance_history: list[np.ndarray] = []

        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            q = state[:4]
            omega = state[4:7]

            # Normalize quaternion
            q = q / np.linalg.norm(q)

            # Compute attitude error
            error_q = self._quaternion_error(q, q_target)
            error_euler = Rotation.from_quat(error_q).as_euler("xyz")

            # PID control
            control_torque = self._pid_control(error_euler, omega, dt)

            # Environmental disturbances
            disturbance = self._compute_disturbances(q, t)

            # Store for later
            control_history.append(control_torque.copy())
            disturbance_history.append(disturbance.copy())

            # Total torque
            total_torque = control_torque + disturbance

            # Euler's equation: I * omega_dot = torque - omega x (I * omega)
            inertia = self.config.inertia_matrix
            omega_cross_h = np.cross(omega, inertia @ omega)
            omega_dot = np.linalg.solve(inertia, total_torque - omega_cross_h)

            # Quaternion kinematics: q_dot = 0.5 * Omega * q
            q_dot = 0.5 * self._omega_matrix(omega) @ q

            return np.concatenate([q_dot, omega_dot])

        logger.info(
            f"Running attitude simulation: {duration_s}s, "
            f"initial={initial_attitude_deg}, target={target_attitude_deg}"
        )

        result = solve_ivp(
            dynamics,
            t_span,
            state0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        # Extract results
        quaternions = result.y[:4].T
        angular_velocities = result.y[4:7].T

        # Normalize quaternions
        for i in range(len(quaternions)):
            quaternions[i] /= np.linalg.norm(quaternions[i])

        # Convert to Euler angles
        euler_angles = np.array(
            [Rotation.from_quat(q).as_euler("xyz", degrees=True) for q in quaternions]
        )

        # Compute pointing errors
        pointing_errors = np.array(
            [
                np.rad2deg(2 * np.arccos(np.clip(abs(np.dot(q, q_target)), 0, 1)))
                for q in quaternions
            ]
        )

        # Pad control/disturbance histories to match time array
        n_steps = len(result.t)
        ctrl_arr = np.zeros((n_steps, 3))
        dist_arr = np.zeros((n_steps, 3))
        n_ctrl = min(len(control_history), n_steps)
        if n_ctrl > 0:
            ctrl_arr[:n_ctrl] = np.array(control_history[:n_ctrl])
            dist_arr[:n_ctrl] = np.array(disturbance_history[:n_ctrl])

        # Compute performance metrics
        settling_time = self._compute_settling_time(result.t, pointing_errors, threshold=1.0)
        max_overshoot = float(np.max(pointing_errors))
        steady_state_error = (
            float(np.mean(pointing_errors[-100:]))
            if len(pointing_errors) > 100
            else float(pointing_errors[-1])
        )

        sim_result = SimulationResult(
            time=result.t,
            quaternions=quaternions,
            angular_velocities=angular_velocities,
            control_torques=ctrl_arr,
            pointing_errors=pointing_errors,
            euler_angles=euler_angles,
            disturbance_torques=dist_arr,
            settling_time=settling_time,
            max_overshoot_deg=max_overshoot,
            steady_state_error_deg=steady_state_error,
        )

        logger.info(
            f"Simulation complete. Settling time: {settling_time:.1f}s, "
            f"Max overshoot: {max_overshoot:.2f} deg"
        )
        return sim_result

    def _pid_control(
        self,
        error_euler: np.ndarray,
        omega: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """PID attitude controller with anti-windup."""
        # Update integral with anti-windup
        self._integral_error += error_euler * dt
        integral_limit = 0.5
        self._integral_error = np.clip(self._integral_error, -integral_limit, integral_limit)

        # PID law
        torque = (
            -self.gains.kp * error_euler
            - self.gains.ki * self._integral_error
            - self.gains.kd * omega
        )

        # Actuator saturation
        max_torque = self.config.rw_max_torque_nm
        torque = np.clip(torque, -max_torque, max_torque)

        return torque

    def _compute_disturbances(self, quaternion: np.ndarray, time: float) -> np.ndarray:
        """Compute environmental disturbance torques."""
        disturbance = np.zeros(3)

        if self.config.gravity_gradient_enabled:
            mu_earth = 3.986e14  # m^3/s^2
            r = (6371 + self.config.orbital_altitude_km) * 1e3
            n = np.sqrt(mu_earth / r**3)  # Mean motion

            rot = Rotation.from_quat(quaternion)
            dcm = rot.as_matrix()
            nadir = dcm @ np.array([0, 0, -1])

            inertia = self.config.inertia_matrix
            gg_torque = 3 * n**2 * np.cross(nadir, inertia @ nadir)
            disturbance += gg_torque

        if self.config.solar_pressure_enabled:
            srp_magnitude = 1e-6  # Typical SRP torque magnitude
            orbital_angle = 2 * np.pi * time / self.config.orbital_period_s
            srp_torque = srp_magnitude * np.array(
                [np.sin(orbital_angle), np.cos(orbital_angle), 0.5 * np.sin(2 * orbital_angle)]
            )
            disturbance += srp_torque

        if self.config.magnetic_torque_enabled:
            mag_magnitude = 5e-7
            disturbance += mag_magnitude * np.array(
                [np.sin(time * 0.001), np.cos(time * 0.001), 0.0]
            )

        return disturbance

    def _quaternion_error(self, q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """Compute quaternion error between current and target attitudes."""
        q_target_inv = q_target.copy()
        q_target_inv[:3] = -q_target_inv[:3]  # Conjugate (invert vector part)
        q_error = self._quaternion_multiply(q_target_inv, q_current)
        if q_error[3] < 0:
            q_error = -q_error
        return q_error

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton quaternion product (scalar-last convention)."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )

    @staticmethod
    def _omega_matrix(omega: np.ndarray) -> np.ndarray:
        """Construct the quaternion rate matrix from angular velocity."""
        wx, wy, wz = omega
        return np.array(
            [
                [0, wz, -wy, wx],
                [-wz, 0, wx, wy],
                [wy, -wx, 0, wz],
                [-wx, -wy, -wz, 0],
            ]
        )

    @staticmethod
    def _compute_settling_time(
        time: np.ndarray,
        error: np.ndarray,
        threshold: float = 1.0,
    ) -> float:
        """Compute settling time (time to reach and stay within threshold)."""
        within_threshold = error < threshold
        for i in range(len(within_threshold) - 1, -1, -1):
            if not within_threshold[i]:
                if i < len(time) - 1:
                    return float(time[i + 1])
                return float(time[-1])
        return 0.0

    def plot_results(
        self,
        result: SimulationResult,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate comprehensive attitude control simulation plots.

        Creates a multi-panel figure showing:
        - Euler angles over time
        - Angular velocities
        - Control torques
        - Pointing error with settling time marker

        Args:
            result: SimulationResult from run_simulation
            output_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Spacecraft Attitude Control Simulation\n(Quaternion PID + Reaction Wheels)",
            fontsize=16,
            fontweight="bold",
        )

        # Panel 1: Euler angles
        ax1 = axes[0, 0]
        labels = ["Roll", "Pitch", "Yaw"]
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax1.plot(
                result.time, result.euler_angles[:, i], label=label, color=color, linewidth=1.5
            )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (degrees)")
        ax1.set_title("Euler Angles")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3)

        # Panel 2: Angular velocities
        ax2 = axes[0, 1]
        rate_labels = ["wx", "wy", "wz"]
        for i, (label, color) in enumerate(zip(rate_labels, colors)):
            ax2.plot(
                result.time,
                np.rad2deg(result.angular_velocities[:, i]),
                label=label,
                color=color,
                linewidth=1.5,
            )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angular Rate (deg/s)")
        ax2.set_title("Angular Velocities")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Control torques
        ax3 = axes[1, 0]
        torque_labels = ["Tx", "Ty", "Tz"]
        for i, (label, color) in enumerate(zip(torque_labels, colors)):
            ax3.plot(
                result.time,
                result.control_torques[:, i] * 1000,  # Convert to mNm
                label=label,
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Torque (mNm)")
        ax3.set_title("Reaction Wheel Control Torques")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Pointing error
        ax4 = axes[1, 1]
        ax4.semilogy(result.time, result.pointing_errors, color="#8e44ad", linewidth=1.5)
        ax4.axhline(y=1.0, color="green", linestyle="--", alpha=0.7, label="1 deg threshold")
        ax4.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="0.1 deg threshold")
        if result.settling_time is not None:
            ax4.axvline(
                x=result.settling_time,
                color="red",
                linestyle=":",
                alpha=0.7,
                label=f"Settling: {result.settling_time:.1f}s",
            )
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Pointing Error (degrees)")
        ax4.set_title("Attitude Pointing Error")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add performance annotation
        perf_text = (
            f"Settling Time: {result.settling_time:.1f} s\n"
            f"Max Overshoot: {result.max_overshoot_deg:.2f} deg\n"
            f"SS Error: {result.steady_state_error_deg:.4f} deg"
        )
        ax4.text(
            0.98,
            0.98,
            perf_text,
            transform=ax4.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Attitude simulation plot saved to {output_path}")

        return fig
