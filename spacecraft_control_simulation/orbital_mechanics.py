"""
Orbital Mechanics Simulation

Two-body and perturbed orbital mechanics simulation with Hohmann transfer
and orbit propagation — key tools for mission design at JPL.

Inspired by:
- nasa-jpl/MonteCop: Trajectory solution interoperability
- JPL's MONTE: Mission Analysis Operations Navigation Toolkit Environment

Provides Keplerian orbit propagation, Hohmann transfer calculation, and
ground track visualization — all tasks that JPL mission designers do daily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# Constants
MU_EARTH = 3.986004418e14  # Earth gravitational parameter (m^3/s^2)
R_EARTH = 6.371e6  # Earth radius (m)
J2 = 1.08263e-3  # Earth J2 oblateness coefficient


@dataclass
class OrbitalElements:
    """Classical orbital elements."""

    semi_major_axis_m: float  # a
    eccentricity: float  # e
    inclination_deg: float  # i
    raan_deg: float  # Omega (Right Ascension of Ascending Node)
    arg_periapsis_deg: float  # omega
    true_anomaly_deg: float  # nu


@dataclass
class HohmannTransfer:
    """Parameters for a Hohmann transfer orbit."""

    delta_v1_ms: float  # First burn (departure)
    delta_v2_ms: float  # Second burn (arrival)
    total_delta_v_ms: float
    transfer_time_s: float
    transfer_semi_major_m: float
    departure_altitude_km: float
    arrival_altitude_km: float


@dataclass
class OrbitPropagationResult:
    """Result of orbit propagation."""

    time: np.ndarray
    position: np.ndarray  # (N, 3) ECI position in meters
    velocity: np.ndarray  # (N, 3) ECI velocity in m/s
    altitude_km: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    orbital_elements_history: list[OrbitalElements]


class OrbitalMechanicsSimulator:
    """
    Orbital mechanics simulation for mission design.

    Provides:
    - Keplerian orbit propagation with J2 perturbation
    - Hohmann transfer orbit calculation
    - Ground track generation
    - Delta-V budget computation
    - Orbit visualization

    Example:
        >>> sim = OrbitalMechanicsSimulator()
        >>> transfer = sim.compute_hohmann_transfer(400, 35786)  # LEO to GEO
        >>> print(f"Total delta-V: {transfer.total_delta_v_ms:.1f} m/s")
        >>> result = sim.propagate_orbit(altitude_km=400, inclination_deg=51.6, periods=3)
        >>> sim.plot_ground_track(result, output_path="ground_track.png")
    """

    def __init__(self, include_j2: bool = True):
        self.include_j2 = include_j2

    def compute_hohmann_transfer(
        self,
        departure_altitude_km: float,
        arrival_altitude_km: float,
    ) -> HohmannTransfer:
        """
        Compute a Hohmann transfer orbit between two circular orbits.

        Args:
            departure_altitude_km: Departure orbit altitude (km above Earth surface)
            arrival_altitude_km: Arrival orbit altitude (km above Earth surface)

        Returns:
            HohmannTransfer with delta-V and timing parameters
        """
        r1 = R_EARTH + departure_altitude_km * 1e3
        r2 = R_EARTH + arrival_altitude_km * 1e3

        # Circular orbit velocities
        v1_circular = np.sqrt(MU_EARTH / r1)
        v2_circular = np.sqrt(MU_EARTH / r2)

        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2

        # Velocities at periapsis and apoapsis of transfer orbit
        v_transfer_periapsis = np.sqrt(MU_EARTH * (2 / r1 - 1 / a_transfer))
        v_transfer_apoapsis = np.sqrt(MU_EARTH * (2 / r2 - 1 / a_transfer))

        # Delta-V for each burn
        delta_v1 = abs(v_transfer_periapsis - v1_circular)
        delta_v2 = abs(v2_circular - v_transfer_apoapsis)

        # Transfer time (half the transfer orbit period)
        transfer_time = np.pi * np.sqrt(a_transfer**3 / MU_EARTH)

        logger.info(
            f"Hohmann transfer: {departure_altitude_km}km -> {arrival_altitude_km}km, "
            f"Total dV={delta_v1 + delta_v2:.1f} m/s, Time={transfer_time / 3600:.2f} hr"
        )

        return HohmannTransfer(
            delta_v1_ms=float(delta_v1),
            delta_v2_ms=float(delta_v2),
            total_delta_v_ms=float(delta_v1 + delta_v2),
            transfer_time_s=float(transfer_time),
            transfer_semi_major_m=float(a_transfer),
            departure_altitude_km=departure_altitude_km,
            arrival_altitude_km=arrival_altitude_km,
        )

    def propagate_orbit(
        self,
        altitude_km: float = 400.0,
        inclination_deg: float = 51.6,
        eccentricity: float = 0.0,
        raan_deg: float = 0.0,
        arg_periapsis_deg: float = 0.0,
        true_anomaly_deg: float = 0.0,
        periods: float = 3.0,
        dt: float = 10.0,
    ) -> OrbitPropagationResult:
        """
        Propagate an orbit using numerical integration.

        Args:
            altitude_km: Orbit altitude (km, for circular orbit)
            inclination_deg: Orbital inclination (degrees)
            eccentricity: Orbital eccentricity
            raan_deg: Right ascension of ascending node (degrees)
            arg_periapsis_deg: Argument of periapsis (degrees)
            true_anomaly_deg: Initial true anomaly (degrees)
            periods: Number of orbital periods to propagate
            dt: Time step (seconds)

        Returns:
            OrbitPropagationResult with full state history
        """
        # Convert orbital elements to state vector
        elements = OrbitalElements(
            semi_major_axis_m=R_EARTH + altitude_km * 1e3,
            eccentricity=eccentricity,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            arg_periapsis_deg=arg_periapsis_deg,
            true_anomaly_deg=true_anomaly_deg,
        )

        r0, v0 = self._elements_to_state(elements)
        state0 = np.concatenate([r0, v0])

        # Orbital period
        a = elements.semi_major_axis_m
        period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        duration = periods * period
        t_eval = np.arange(0, duration, dt)

        logger.info(
            f"Propagating orbit: alt={altitude_km}km, inc={inclination_deg}deg, "
            f"{periods} periods ({duration / 3600:.1f} hr)"
        )

        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            r = state[:3]
            v = state[3:6]
            r_norm = np.linalg.norm(r)

            # Two-body acceleration
            a_grav = -MU_EARTH * r / r_norm**3

            # J2 perturbation
            if self.include_j2:
                z = r[2]
                factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_norm**5
                a_j2 = np.array(
                    [
                        factor * r[0] * (5 * z**2 / r_norm**2 - 1),
                        factor * r[1] * (5 * z**2 / r_norm**2 - 1),
                        factor * r[2] * (5 * z**2 / r_norm**2 - 3),
                    ]
                )
                a_grav += a_j2

            return np.concatenate([v, a_grav])

        result = solve_ivp(
            dynamics,
            (0, duration),
            state0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )

        positions = result.y[:3].T
        velocities = result.y[3:6].T

        # Compute derived quantities
        altitudes = np.linalg.norm(positions, axis=1) / 1e3 - R_EARTH / 1e3
        lats, lons = self._eci_to_ground_track(positions, result.t)

        # Sample orbital elements history
        elements_history = []
        sample_indices = np.linspace(0, len(result.t) - 1, min(100, len(result.t)), dtype=int)
        for idx in sample_indices:
            oe = self._state_to_elements(positions[idx], velocities[idx])
            elements_history.append(oe)

        return OrbitPropagationResult(
            time=result.t,
            position=positions,
            velocity=velocities,
            altitude_km=altitudes,
            latitude_deg=lats,
            longitude_deg=lons,
            orbital_elements_history=elements_history,
        )

    def plot_ground_track(
        self,
        result: OrbitPropagationResult,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate ground track and orbit visualization.

        Creates a multi-panel figure showing:
        - Ground track (latitude/longitude)
        - 3D orbit visualization
        - Altitude profile
        - Orbital elements evolution

        Args:
            result: OrbitPropagationResult from propagate_orbit
            output_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Orbital Mechanics Simulation\n(Keplerian + J2 Perturbation)",
            fontsize=16,
            fontweight="bold",
        )

        # Panel 1: Ground track
        ax1 = axes[0, 0]
        ax1.scatter(
            result.longitude_deg,
            result.latitude_deg,
            c=result.time / 3600,
            cmap="viridis",
            s=1,
            alpha=0.5,
        )
        ax1.set_xlim(-180, 180)
        ax1.set_ylim(-90, 90)
        ax1.set_xlabel("Longitude (deg)")
        ax1.set_ylabel("Latitude (deg)")
        ax1.set_title("Ground Track")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        # Panel 2: 3D orbit (projected to 2D)
        ax2 = axes[0, 1]
        pos_km = result.position / 1e3
        ax2.plot(pos_km[:, 0], pos_km[:, 1], linewidth=0.5, alpha=0.7, color="steelblue")
        theta = np.linspace(0, 2 * np.pi, 100)
        ax2.plot(
            R_EARTH / 1e3 * np.cos(theta),
            R_EARTH / 1e3 * np.sin(theta),
            "g-",
            linewidth=2,
            label="Earth",
        )
        ax2.set_xlabel("X (km)")
        ax2.set_ylabel("Y (km)")
        ax2.set_title("Orbit (XY Projection)")
        ax2.set_aspect("equal")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Altitude profile
        ax3 = axes[1, 0]
        ax3.plot(result.time / 3600, result.altitude_km, color="#e74c3c", linewidth=1)
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Altitude (km)")
        ax3.set_title("Altitude Profile")
        ax3.grid(True, alpha=0.3)

        # Panel 4: Hohmann transfer comparison table
        ax4 = axes[1, 1]
        ax4.axis("off")
        transfers = [
            ("LEO → MEO", 400, 20200),
            ("LEO → GEO", 400, 35786),
            ("LEO → Lunar", 400, 384400),
        ]

        table_data = [["Transfer", "ΔV₁ (m/s)", "ΔV₂ (m/s)", "Total ΔV", "Time (hr)"]]
        for name, dep, arr in transfers:
            try:
                ht = self.compute_hohmann_transfer(dep, arr)
                table_data.append(
                    [
                        name,
                        f"{ht.delta_v1_ms:.0f}",
                        f"{ht.delta_v2_ms:.0f}",
                        f"{ht.total_delta_v_ms:.0f}",
                        f"{ht.transfer_time_s / 3600:.1f}",
                    ]
                )
            except Exception:
                continue

        table = ax4.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title("Hohmann Transfer Comparison", pad=20)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Orbital mechanics plot saved to {output_path}")

        return fig

    def _elements_to_state(self, elements: OrbitalElements) -> tuple[np.ndarray, np.ndarray]:
        """Convert orbital elements to ECI state vector (position, velocity)."""
        a = elements.semi_major_axis_m
        e = elements.eccentricity
        i = np.deg2rad(elements.inclination_deg)
        raan = np.deg2rad(elements.raan_deg)
        omega = np.deg2rad(elements.arg_periapsis_deg)
        nu = np.deg2rad(elements.true_anomaly_deg)

        # Position and velocity in perifocal frame
        p = a * (1 - e**2)
        r_pf = p / (1 + e * np.cos(nu))
        r_vec_pf = r_pf * np.array([np.cos(nu), np.sin(nu), 0])
        v_vec_pf = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

        # Rotation matrix from perifocal to ECI
        cos_raan, sin_raan = np.cos(raan), np.sin(raan)
        cos_omega, sin_omega = np.cos(omega), np.sin(omega)
        cos_i, sin_i = np.cos(i), np.sin(i)

        rot = np.array(
            [
                [
                    cos_raan * cos_omega - sin_raan * sin_omega * cos_i,
                    -cos_raan * sin_omega - sin_raan * cos_omega * cos_i,
                    sin_raan * sin_i,
                ],
                [
                    sin_raan * cos_omega + cos_raan * sin_omega * cos_i,
                    -sin_raan * sin_omega + cos_raan * cos_omega * cos_i,
                    -cos_raan * sin_i,
                ],
                [sin_omega * sin_i, cos_omega * sin_i, cos_i],
            ]
        )

        r_eci = rot @ r_vec_pf
        v_eci = rot @ v_vec_pf

        return r_eci, v_eci

    def _state_to_elements(self, r: np.ndarray, v: np.ndarray) -> OrbitalElements:
        """Convert ECI state vector to orbital elements."""
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)

        # Specific angular momentum
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)

        # Node vector
        k_hat = np.array([0, 0, 1])
        n = np.cross(k_hat, h)
        n_norm = np.linalg.norm(n)

        # Eccentricity vector
        e_vec = ((v_norm**2 - MU_EARTH / r_norm) * r - np.dot(r, v) * v) / MU_EARTH
        e = np.linalg.norm(e_vec)

        # Semi-major axis
        energy = v_norm**2 / 2 - MU_EARTH / r_norm
        if abs(energy) > 1e-10:
            a = -MU_EARTH / (2 * energy)
        else:
            a = float("inf")

        # Inclination
        inc = np.arccos(np.clip(h[2] / h_norm, -1, 1))

        # RAAN
        if n_norm > 1e-10:
            raan = np.arccos(np.clip(n[0] / n_norm, -1, 1))
            if n[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0.0

        # Argument of periapsis
        if n_norm > 1e-10 and e > 1e-10:
            omega = np.arccos(np.clip(np.dot(n, e_vec) / (n_norm * e), -1, 1))
            if e_vec[2] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0.0

        # True anomaly
        if e > 1e-10:
            nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_norm), -1, 1))
            if np.dot(r, v) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0.0

        return OrbitalElements(
            semi_major_axis_m=float(a),
            eccentricity=float(e),
            inclination_deg=float(np.rad2deg(inc)),
            raan_deg=float(np.rad2deg(raan)),
            arg_periapsis_deg=float(np.rad2deg(omega)),
            true_anomaly_deg=float(np.rad2deg(nu)),
        )

    @staticmethod
    def _eci_to_ground_track(
        positions: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert ECI positions to ground track (lat, lon) accounting for Earth rotation."""
        omega_earth = 7.2921159e-5  # Earth rotation rate (rad/s)

        lats = np.zeros(len(positions))
        lons = np.zeros(len(positions))

        for i, (pos, t) in enumerate(zip(positions, times)):
            r_norm = np.linalg.norm(pos)
            lat = np.arcsin(np.clip(pos[2] / r_norm, -1, 1))

            # Account for Earth rotation
            theta = np.arctan2(pos[1], pos[0]) - omega_earth * t
            lon = theta % (2 * np.pi)
            if lon > np.pi:
                lon -= 2 * np.pi

            lats[i] = np.rad2deg(lat)
            lons[i] = np.rad2deg(lon)

        return lats, lons
