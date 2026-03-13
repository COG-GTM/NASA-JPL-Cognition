"""
Spacecraft Trajectory Uncertainty Quantification via Monte Carlo

Implements Monte Carlo methods for quantifying trajectory dispersions
in interplanetary and orbital missions — a core JPL mission design task.

Inspired by:
- nasa/SMCPy: Sequential Monte Carlo with Python
- nasa-jpl/MonteCop: Monte/Copernicus trajectory tool interoperability
- JPL's MONTE: Mission-Analysis Operations Navigation Toolkit Environment

At JPL, trajectory Monte Carlo runs are used for:
- Launch window analysis
- Orbit insertion uncertainty
- Entry, Descent, and Landing (EDL) dispersion analysis
- Navigation accuracy assessment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryParameters:
    """Nominal trajectory parameters with uncertainties (1-sigma)."""

    # Initial state uncertainties
    position_uncertainty_km: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.5]))
    velocity_uncertainty_ms: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.005])
    )

    # Maneuver execution errors
    delta_v_magnitude_error_pct: float = 0.5  # 0.5% 1-sigma
    delta_v_pointing_error_deg: float = 0.1  # 0.1 deg 1-sigma

    # Atmospheric entry parameters (for EDL analysis)
    entry_flight_path_angle_deg: float = -15.5
    entry_fpa_uncertainty_deg: float = 0.1
    entry_velocity_kms: float = 5.5
    entry_velocity_uncertainty_kms: float = 0.005

    # Atmospheric model uncertainty
    density_scale_factor: float = 1.0
    density_uncertainty_pct: float = 10.0

    # Gravity model errors
    gravity_uncertainty_pct: float = 0.01

    # Target parameters
    target_altitude_km: float = 400.0
    target_latitude_deg: float = 0.0
    target_longitude_deg: float = 0.0


@dataclass
class MonteCarloResult:
    """Results from a trajectory Monte Carlo analysis."""

    n_samples: int
    landing_latitude_deg: np.ndarray
    landing_longitude_deg: np.ndarray
    landing_altitude_km: np.ndarray
    time_of_flight_s: np.ndarray
    max_deceleration_g: np.ndarray
    final_velocity_ms: np.ndarray
    success_rate: float
    percentile_99_ellipse_km: tuple[float, float]
    mean_landing_error_km: float
    covariance_matrix: np.ndarray


class TrajectoryUncertaintyAnalyzer:
    """
    Monte Carlo trajectory uncertainty analysis.

    Runs thousands of trajectory simulations with perturbed initial
    conditions to quantify landing/arrival dispersions — the same
    methodology JPL uses for Mars EDL analysis.

    Example:
        >>> analyzer = TrajectoryUncertaintyAnalyzer(n_samples=10000)
        >>> params = TrajectoryParameters()
        >>> result = analyzer.run_edl_monte_carlo(params)
        >>> print(f"99% landing ellipse: {result.percentile_99_ellipse_km}")
        >>> analyzer.plot_results(result, output_path="edl_monte_carlo.png")
    """

    def __init__(self, n_samples: int = 5000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def run_edl_monte_carlo(
        self,
        params: Optional[TrajectoryParameters] = None,
    ) -> MonteCarloResult:
        """
        Run Entry, Descent, and Landing Monte Carlo dispersion analysis.

        Simulates atmospheric entry trajectories with perturbed initial
        conditions and atmospheric models to quantify landing dispersions.

        This is a simplified 3-DOF model of what JPL runs with their
        full 6-DOF POST/DSENDS simulators for Mars missions.

        Args:
            params: Trajectory parameters with uncertainties

        Returns:
            MonteCarloResult with dispersion statistics
        """
        if params is None:
            params = TrajectoryParameters()

        logger.info(f"Running EDL Monte Carlo with {self.n_samples} samples")

        # Pre-allocate result arrays
        landing_lats = np.zeros(self.n_samples)
        landing_lons = np.zeros(self.n_samples)
        landing_alts = np.zeros(self.n_samples)
        tof_values = np.zeros(self.n_samples)
        max_decel = np.zeros(self.n_samples)
        final_velocities = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            # Perturb entry conditions
            fpa = params.entry_flight_path_angle_deg + (
                self.rng.standard_normal() * params.entry_fpa_uncertainty_deg
            )
            v_entry = params.entry_velocity_kms + (
                self.rng.standard_normal() * params.entry_velocity_uncertainty_kms
            )
            density_factor = params.density_scale_factor + (
                self.rng.standard_normal() * params.density_uncertainty_pct / 100.0
            )

            # Position perturbation
            pos_pert = self.rng.standard_normal(3) * params.position_uncertainty_km
            vel_pert = self.rng.standard_normal(3) * params.velocity_uncertainty_ms / 1000.0

            # Simplified 3-DOF trajectory propagation
            result = self._propagate_entry_trajectory(
                fpa_deg=fpa,
                velocity_kms=v_entry,
                density_factor=density_factor,
                position_offset_km=pos_pert,
                velocity_offset_kms=vel_pert,
                target_lat=params.target_latitude_deg,
                target_lon=params.target_longitude_deg,
            )

            landing_lats[i] = result["latitude_deg"]
            landing_lons[i] = result["longitude_deg"]
            landing_alts[i] = result["altitude_km"]
            tof_values[i] = result["time_of_flight_s"]
            max_decel[i] = result["max_deceleration_g"]
            final_velocities[i] = result["final_velocity_ms"]

        # Compute statistics
        lat_errors = landing_lats - params.target_latitude_deg
        lon_errors = landing_lons - params.target_longitude_deg

        # Convert to km (approximate)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.deg2rad(params.target_latitude_deg))

        x_errors_km = lon_errors * km_per_deg_lon
        y_errors_km = lat_errors * km_per_deg_lat

        mean_error_km = float(np.mean(np.sqrt(x_errors_km**2 + y_errors_km**2)))

        # 99% landing ellipse (2D Gaussian)
        cov_matrix = np.cov(x_errors_km, y_errors_km)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        chi2_99 = stats.chi2.ppf(0.99, 2)
        ellipse_semi_axes = np.sqrt(eigenvalues * chi2_99)

        success_mask = (final_velocities < 5.0) & (landing_alts < 1.0)
        success_rate = float(np.mean(success_mask))

        logger.info(
            f"EDL Monte Carlo complete: "
            f"99%% ellipse = {ellipse_semi_axes[0]:.2f} x {ellipse_semi_axes[1]:.2f} km, "
            f"Success rate = {success_rate * 100:.1f}%%"
        )

        return MonteCarloResult(
            n_samples=self.n_samples,
            landing_latitude_deg=landing_lats,
            landing_longitude_deg=landing_lons,
            landing_altitude_km=landing_alts,
            time_of_flight_s=tof_values,
            max_deceleration_g=max_decel,
            final_velocity_ms=final_velocities,
            success_rate=success_rate,
            percentile_99_ellipse_km=(
                float(ellipse_semi_axes[0]),
                float(ellipse_semi_axes[1]),
            ),
            mean_landing_error_km=mean_error_km,
            covariance_matrix=cov_matrix,
        )

    def run_orbit_insertion_monte_carlo(
        self,
        nominal_delta_v_ms: float = 1000.0,
        delta_v_error_pct: float = 0.5,
        pointing_error_deg: float = 0.1,
        target_altitude_km: float = 400.0,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo for orbit insertion burn uncertainty.

        Simulates the effect of thrust magnitude and pointing errors
        on the resulting orbit after an insertion burn.

        Args:
            nominal_delta_v_ms: Nominal delta-V in m/s
            delta_v_error_pct: 1-sigma magnitude error (%)
            pointing_error_deg: 1-sigma pointing error (degrees)
            target_altitude_km: Target orbit altitude

        Returns:
            MonteCarloResult with orbit dispersions
        """
        logger.info(f"Running orbit insertion Monte Carlo ({self.n_samples} samples)")

        landing_lats = np.zeros(self.n_samples)
        landing_lons = np.zeros(self.n_samples)
        altitudes = np.zeros(self.n_samples)
        tof_values = np.zeros(self.n_samples)
        max_decel = np.zeros(self.n_samples)
        final_velocities = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            # Perturb delta-V magnitude
            dv_actual = nominal_delta_v_ms * (
                1.0 + self.rng.standard_normal() * delta_v_error_pct / 100.0
            )

            # Perturb pointing
            pointing_err_rad = np.deg2rad(self.rng.standard_normal() * pointing_error_deg)

            # Compute resulting orbit altitude deviation
            dv_along_track = dv_actual * np.cos(pointing_err_rad)
            dv_cross_track = dv_actual * np.sin(pointing_err_rad)

            # Simplified: altitude change proportional to dV error
            dv_error = dv_along_track - nominal_delta_v_ms
            altitude_error = dv_error * 0.5  # ~0.5 km per m/s (simplified)
            cross_track_error = dv_cross_track * 0.3  # Cross-track displacement

            altitudes[i] = target_altitude_km + altitude_error
            landing_lats[i] = cross_track_error / 111.32  # Convert km to deg
            landing_lons[i] = altitude_error / 111.32
            final_velocities[i] = abs(dv_error)
            tof_values[i] = 0.0
            max_decel[i] = 0.0

        cov_matrix = np.cov(landing_lats * 111.32, landing_lons * 111.32)
        mean_alt_error = float(np.mean(np.abs(altitudes - target_altitude_km)))

        return MonteCarloResult(
            n_samples=self.n_samples,
            landing_latitude_deg=landing_lats,
            landing_longitude_deg=landing_lons,
            landing_altitude_km=altitudes,
            time_of_flight_s=tof_values,
            max_deceleration_g=max_decel,
            final_velocity_ms=final_velocities,
            success_rate=float(np.mean(np.abs(altitudes - target_altitude_km) < 10.0)),
            percentile_99_ellipse_km=(mean_alt_error * 2.576, mean_alt_error * 2.576),
            mean_landing_error_km=mean_alt_error,
            covariance_matrix=cov_matrix,
        )

    def _propagate_entry_trajectory(
        self,
        fpa_deg: float,
        velocity_kms: float,
        density_factor: float,
        position_offset_km: np.ndarray,
        velocity_offset_kms: np.ndarray,
        target_lat: float,
        target_lon: float,
    ) -> dict[str, float]:
        """
        Simplified 3-DOF atmospheric entry trajectory propagation.

        Uses ballistic coefficient and exponential atmosphere model.
        """
        # Entry parameters
        beta = 100.0  # Ballistic coefficient (kg/m^2) — typical Mars lander
        r_planet = 3389.5  # Mars radius (km)
        g_surface = 3.721  # Mars surface gravity (m/s^2)
        scale_height = 11.1  # Atmospheric scale height (km)

        # Initial conditions
        h = 125.0  # Entry interface altitude (km)
        v = velocity_kms
        gamma = np.deg2rad(fpa_deg)

        dt = 0.5  # Time step (seconds)
        t = 0.0
        max_g = 0.0

        # Propagate until landing
        while h > 0 and t < 600:
            # Atmospheric density (exponential model)
            rho = 0.020 * density_factor * np.exp(-h / scale_height)

            # Drag deceleration
            drag_decel = 0.5 * rho * (v * 1000) ** 2 / beta / 1000  # km/s^2

            # Gravity component
            g_local = g_surface * (r_planet / (r_planet + h)) ** 2

            # Equations of motion
            dv_dt = -drag_decel - g_local * np.sin(gamma) / 1000
            dgamma_dt = (
                (1.0 / v)
                * ((v**2 / (r_planet + h)) * np.cos(gamma) - g_local * np.cos(gamma) / 1000)
                if v > 0.001
                else 0
            )

            dh_dt = v * np.sin(gamma)

            # Update state
            v += dv_dt * dt
            gamma += dgamma_dt * dt
            h += dh_dt * dt
            t += dt

            # Track max deceleration
            current_g = abs(drag_decel * 1000 / g_surface)
            if current_g > max_g:
                max_g = current_g

            if v < 0.001:
                break

        # Landing location with perturbations
        downrange_km = velocity_kms * t * np.cos(np.deg2rad(fpa_deg)) * 0.1
        crossrange_km = position_offset_km[1] + velocity_offset_kms[1] * t

        lat_offset = crossrange_km / 111.32
        lon_offset = downrange_km / (111.32 * np.cos(np.deg2rad(target_lat)))

        return {
            "latitude_deg": target_lat + lat_offset + self.rng.standard_normal() * 0.01,
            "longitude_deg": target_lon + lon_offset + self.rng.standard_normal() * 0.01,
            "altitude_km": max(0, h),
            "time_of_flight_s": t,
            "max_deceleration_g": max_g,
            "final_velocity_ms": max(0, v * 1000),
        }

    def plot_results(
        self,
        result: MonteCarloResult,
        output_path: Optional[str] = None,
        title: str = "Entry, Descent & Landing Monte Carlo Analysis",
    ) -> plt.Figure:
        """
        Generate comprehensive Monte Carlo dispersion visualization.

        Creates a multi-panel figure showing:
        - Landing footprint scatter plot with 99% ellipse
        - Landing error histogram
        - Max deceleration distribution
        - Convergence plot

        Args:
            result: MonteCarloResult from run_edl_monte_carlo
            output_path: Optional path to save figure
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"{title}\n({result.n_samples:,} samples)",
            fontsize=16,
            fontweight="bold",
        )

        # Panel 1: Landing footprint
        ax1 = axes[0, 0]
        km_per_deg = 111.32
        x_km = (result.landing_longitude_deg - np.mean(result.landing_longitude_deg)) * km_per_deg
        y_km = (result.landing_latitude_deg - np.mean(result.landing_latitude_deg)) * km_per_deg

        ax1.scatter(x_km, y_km, s=1, alpha=0.3, c="steelblue")
        ax1.scatter(0, 0, s=100, c="red", marker="x", zorder=5, label="Target")

        # Draw 99% ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(result.covariance_matrix)
        chi2_99 = stats.chi2.ppf(0.99, 2)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        width = 2 * np.sqrt(eigenvalues[1] * chi2_99)
        height = 2 * np.sqrt(eigenvalues[0] * chi2_99)

        from matplotlib.patches import Ellipse

        ellipse = Ellipse(
            (0, 0),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            color="red",
            linewidth=2,
            label=f"99% ellipse ({width:.1f}×{height:.1f} km)",
        )
        ax1.add_patch(ellipse)
        ax1.set_xlabel("Downrange (km)")
        ax1.set_ylabel("Crossrange (km)")
        ax1.set_title("Landing Footprint Dispersion")
        ax1.legend(fontsize=9)
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Landing error histogram
        ax2 = axes[0, 1]
        errors_km = np.sqrt(x_km**2 + y_km**2)
        ax2.hist(errors_km, bins=50, color="steelblue", edgecolor="black", alpha=0.7, density=True)
        ax2.axvline(
            x=np.percentile(errors_km, 99),
            color="red",
            linestyle="--",
            label=f"99th percentile: {np.percentile(errors_km, 99):.2f} km",
        )
        ax2.axvline(
            x=np.mean(errors_km),
            color="green",
            linestyle="--",
            label=f"Mean: {np.mean(errors_km):.2f} km",
        )
        ax2.set_xlabel("Landing Error (km)")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("Landing Error Distribution")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Max deceleration distribution
        ax3 = axes[1, 0]
        valid_decel = result.max_deceleration_g[result.max_deceleration_g > 0]
        if len(valid_decel) > 0:
            ax3.hist(
                valid_decel,
                bins=50,
                color="#e74c3c",
                edgecolor="black",
                alpha=0.7,
                density=True,
            )
            ax3.axvline(
                x=np.percentile(valid_decel, 99),
                color="black",
                linestyle="--",
                label=f"99th: {np.percentile(valid_decel, 99):.1f} g",
            )
        ax3.set_xlabel("Max Deceleration (g)")
        ax3.set_ylabel("Probability Density")
        ax3.set_title("Peak Deceleration Distribution")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Convergence plot
        ax4 = axes[1, 1]
        sample_sizes = np.arange(100, result.n_samples, 100)
        running_means = [np.mean(errors_km[:n]) for n in sample_sizes]
        running_stds = [np.std(errors_km[:n]) for n in sample_sizes]

        ax4.plot(sample_sizes, running_means, color="steelblue", label="Mean Error")
        ax4.fill_between(
            sample_sizes,
            np.array(running_means) - np.array(running_stds),
            np.array(running_means) + np.array(running_stds),
            alpha=0.2,
            color="steelblue",
        )
        ax4.set_xlabel("Number of Samples")
        ax4.set_ylabel("Mean Landing Error (km)")
        ax4.set_title("Monte Carlo Convergence")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Monte Carlo plot saved to {output_path}")

        return fig
