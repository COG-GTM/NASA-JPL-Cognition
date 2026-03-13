"""
Radiation Environment Monte Carlo Modeling

Models the space radiation environment and its effects on spacecraft
electronics — a critical analysis for every JPL deep-space mission.

Simulates:
- Galactic Cosmic Ray (GCR) flux variation with solar cycle
- Solar Particle Events (SPE) probabilistic occurrence
- Trapped radiation belt dose (Van Allen belts)
- Total Ionizing Dose (TID) accumulation
- Single Event Effect (SEE) rate estimation

Reference: JPL Design Principles (JPL D-17868) and ECSS-E-ST-10-12C
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RadiationEnvironmentConfig:
    """Configuration for radiation environment modeling."""

    # Mission parameters
    mission_duration_years: float = 5.0
    shielding_thickness_mm_al: float = 3.0  # Aluminum equivalent shielding

    # Orbit/trajectory
    environment_type: str = "interplanetary"  # "leo", "geo", "interplanetary", "jupiter"
    altitude_km: float = 0.0  # For LEO/GEO
    inclination_deg: float = 0.0

    # Solar cycle phase (0 = solar minimum, 1 = solar maximum)
    solar_cycle_phase: float = 0.5

    # Component radiation tolerance
    tid_limit_krad: float = 100.0  # Total Ionizing Dose limit
    see_cross_section_cm2: float = 1e-8  # SEE cross-section
    displacement_damage_limit: float = 1e12  # Displacement damage dose limit


@dataclass
class RadiationAnalysisResult:
    """Results from radiation environment Monte Carlo analysis."""

    n_samples: int
    total_dose_krad: np.ndarray
    dose_rate_krad_per_year: np.ndarray
    n_see_events: np.ndarray
    spe_fluence: np.ndarray
    gcr_dose_krad: np.ndarray
    trapped_dose_krad: np.ndarray
    spe_dose_krad: np.ndarray
    tid_margin: float  # Positive = within limit
    see_rate_per_day: float
    probability_exceeding_tid_limit: float
    mean_total_dose_krad: float
    dose_99_percentile_krad: float


class RadiationEnvironmentModel:
    """
    Space radiation environment Monte Carlo analysis.

    Models the stochastic nature of the space radiation environment
    to predict total ionizing dose (TID), single event effects (SEE),
    and displacement damage over a mission lifetime.

    Example:
        >>> model = RadiationEnvironmentModel(n_samples=10000)
        >>> config = RadiationEnvironmentConfig(
        ...     mission_duration_years=6.0,
        ...     shielding_thickness_mm_al=3.0,
        ...     environment_type="jupiter",
        ... )
        >>> result = model.run_analysis(config)
        >>> print(f"Mean TID: {result.mean_total_dose_krad:.1f} krad")
        >>> model.plot_results(result, output_path="radiation.png")
    """

    def __init__(self, n_samples: int = 10000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def run_analysis(
        self,
        config: Optional[RadiationEnvironmentConfig] = None,
    ) -> RadiationAnalysisResult:
        """
        Run radiation environment Monte Carlo analysis.

        Args:
            config: Radiation environment configuration

        Returns:
            RadiationAnalysisResult with dose and SEE statistics
        """
        if config is None:
            config = RadiationEnvironmentConfig()

        logger.info(
            f"Running radiation analysis: {config.environment_type}, "
            f"{config.mission_duration_years} years, "
            f"{config.shielding_thickness_mm_al}mm Al shielding"
        )

        # Shielding attenuation factor (simplified exponential model)
        attenuation = self._compute_shielding_attenuation(config.shielding_thickness_mm_al)

        gcr_doses = np.zeros(self.n_samples)
        trapped_doses = np.zeros(self.n_samples)
        spe_doses = np.zeros(self.n_samples)
        see_counts = np.zeros(self.n_samples, dtype=int)
        spe_fluences = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            # GCR dose (varies with solar cycle)
            gcr_base = self._get_gcr_dose_rate(config)
            gcr_variation = self.rng.lognormal(0, 0.15)
            gcr_doses[i] = gcr_base * gcr_variation * config.mission_duration_years * attenuation

            # Trapped radiation (for LEO/GEO)
            trapped_base = self._get_trapped_dose_rate(config)
            trapped_variation = self.rng.lognormal(0, 0.1)
            trapped_doses[i] = (
                trapped_base * trapped_variation * config.mission_duration_years * attenuation
            )

            # Solar Particle Events (stochastic)
            spe_result = self._simulate_spe_events(config, attenuation)
            spe_doses[i] = spe_result["dose_krad"]
            spe_fluences[i] = spe_result["fluence"]

            # Single Event Effects
            total_flux = self._get_particle_flux(config)
            see_rate = (
                total_flux
                * config.see_cross_section_cm2
                * config.mission_duration_years
                * 365.25
                * 24
                * 3600
            )
            see_counts[i] = self.rng.poisson(see_rate)

        # Compute total dose
        total_doses = gcr_doses + trapped_doses + spe_doses
        dose_rates = total_doses / config.mission_duration_years

        # Statistics
        mean_dose = float(np.mean(total_doses))
        dose_99 = float(np.percentile(total_doses, 99))
        tid_margin = config.tid_limit_krad - dose_99
        prob_exceed = float(np.mean(total_doses > config.tid_limit_krad))
        see_rate_per_day = float(np.mean(see_counts)) / (config.mission_duration_years * 365.25)

        logger.info(
            f"Radiation analysis complete: Mean TID={mean_dose:.1f} krad, "
            f"99th={dose_99:.1f} krad, TID margin={tid_margin:.1f} krad"
        )

        return RadiationAnalysisResult(
            n_samples=self.n_samples,
            total_dose_krad=total_doses,
            dose_rate_krad_per_year=dose_rates,
            n_see_events=see_counts,
            spe_fluence=spe_fluences,
            gcr_dose_krad=gcr_doses,
            trapped_dose_krad=trapped_doses,
            spe_dose_krad=spe_doses,
            tid_margin=tid_margin,
            see_rate_per_day=see_rate_per_day,
            probability_exceeding_tid_limit=prob_exceed,
            mean_total_dose_krad=mean_dose,
            dose_99_percentile_krad=dose_99,
        )

    def _compute_shielding_attenuation(self, thickness_mm: float) -> float:
        """Compute radiation attenuation through aluminum shielding."""
        # Simplified exponential attenuation model
        # Real calculations use SHIELDOSE-2 or MULASSIS
        half_value_thickness = 8.0  # mm Al for typical spectrum
        return float(0.5 ** (thickness_mm / half_value_thickness))

    def _get_gcr_dose_rate(self, config: RadiationEnvironmentConfig) -> float:
        """Get Galactic Cosmic Ray dose rate (krad/year)."""
        # GCR is anti-correlated with solar activity
        base_rates = {
            "leo": 0.05,
            "geo": 0.15,
            "interplanetary": 0.10,
            "jupiter": 0.08,  # Lower GCR behind Jupiter's magnetosphere
        }
        base = base_rates.get(config.environment_type, 0.10)
        # Solar minimum = higher GCR, solar maximum = lower GCR
        solar_modulation = 1.5 - 0.5 * config.solar_cycle_phase
        return base * solar_modulation

    def _get_trapped_dose_rate(self, config: RadiationEnvironmentConfig) -> float:
        """Get trapped radiation belt dose rate (krad/year)."""
        rates = {
            "leo": 1.0 if config.inclination_deg > 50 else 0.1,
            "geo": 5.0,
            "interplanetary": 0.0,  # No trapped belts
            "jupiter": 50.0,  # Europa Clipper-relevant!
        }
        return rates.get(config.environment_type, 0.0)

    def _simulate_spe_events(
        self, config: RadiationEnvironmentConfig, attenuation: float
    ) -> dict[str, float]:
        """Simulate Solar Particle Events over mission duration."""
        # SPE rate depends on solar cycle (higher during solar max)
        spe_rate_per_year = 2.0 * config.solar_cycle_phase + 0.5
        expected_events = spe_rate_per_year * config.mission_duration_years
        n_events = self.rng.poisson(expected_events)

        total_dose = 0.0
        total_fluence = 0.0

        for _ in range(n_events):
            # SPE fluence follows log-normal distribution
            # Based on JPL-92-1 model
            log_fluence = self.rng.normal(9.0, 1.5)  # log10(protons/cm^2)
            fluence = 10**log_fluence
            total_fluence += fluence

            # Dose from this event (simplified)
            dose = fluence * 1e-10 * attenuation  # Very simplified dose conversion
            total_dose += dose

        return {"dose_krad": total_dose, "fluence": total_fluence}

    def _get_particle_flux(self, config: RadiationEnvironmentConfig) -> float:
        """Get heavy-ion particle flux for SEE calculation (particles/cm^2/s)."""
        fluxes = {
            "leo": 1e-4,
            "geo": 5e-4,
            "interplanetary": 3e-4,
            "jupiter": 1e-2,  # Intense trapped particle environment
        }
        return fluxes.get(config.environment_type, 3e-4)

    def plot_results(
        self,
        result: RadiationAnalysisResult,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate radiation environment analysis visualization.

        Args:
            result: RadiationAnalysisResult
            output_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Space Radiation Environment Monte Carlo Analysis\n({result.n_samples:,} samples)",
            fontsize=16,
            fontweight="bold",
        )

        # Panel 1: Total dose distribution
        ax1 = axes[0, 0]
        ax1.hist(
            result.total_dose_krad,
            bins=50,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
            density=True,
        )
        ax1.axvline(
            x=result.mean_total_dose_krad,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {result.mean_total_dose_krad:.1f} krad",
        )
        ax1.axvline(
            x=result.dose_99_percentile_krad,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"99th: {result.dose_99_percentile_krad:.1f} krad",
        )
        ax1.set_xlabel("Total Ionizing Dose (krad)")
        ax1.set_ylabel("Probability Density")
        ax1.set_title("Total Dose Distribution (TID)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Dose breakdown by source
        ax2 = axes[0, 1]
        sources = ["GCR", "Trapped", "SPE"]
        mean_doses = [
            float(np.mean(result.gcr_dose_krad)),
            float(np.mean(result.trapped_dose_krad)),
            float(np.mean(result.spe_dose_krad)),
        ]
        std_doses = [
            float(np.std(result.gcr_dose_krad)),
            float(np.std(result.trapped_dose_krad)),
            float(np.std(result.spe_dose_krad)),
        ]
        x = np.arange(len(sources))
        bars = ax2.bar(
            x,
            mean_doses,
            yerr=std_doses,
            color=["#3498db", "#e74c3c", "#f39c12"],
            edgecolor="black",
            capsize=5,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(sources)
        ax2.set_ylabel("Dose (krad)")
        ax2.set_title("Dose Breakdown by Source")
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, dose in zip(bars, mean_doses):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(std_doses) * 0.1,
                f"{dose:.2f}",
                ha="center",
                fontsize=10,
            )

        # Panel 3: SEE event distribution
        ax3 = axes[1, 0]
        ax3.hist(
            result.n_see_events,
            bins=range(0, int(np.max(result.n_see_events)) + 2),
            color="#9b59b6",
            edgecolor="black",
            alpha=0.7,
            density=True,
        )
        ax3.set_xlabel("Number of Single Event Effects")
        ax3.set_ylabel("Probability")
        ax3.set_title(f"SEE Distribution (mean rate: {result.see_rate_per_day:.3f}/day)")
        ax3.grid(True, alpha=0.3)

        # Panel 4: Summary
        ax4 = axes[1, 1]
        ax4.axis("off")
        margin_status = "PASS" if result.tid_margin > 0 else "FAIL"
        margin_color = "#e8f8e8" if result.tid_margin > 0 else "#f8e8e8"

        summary = (
            f"RADIATION ANALYSIS SUMMARY\n"
            f"{'=' * 40}\n\n"
            f"Mean Total Dose: {result.mean_total_dose_krad:.2f} krad\n"
            f"99th Percentile: {result.dose_99_percentile_krad:.2f} krad\n"
            f"TID Margin (99%): {result.tid_margin:.2f} krad [{margin_status}]\n\n"
            f"P(exceeding TID limit): {result.probability_exceeding_tid_limit:.3%}\n\n"
            f"SEE Rate: {result.see_rate_per_day:.4f} events/day\n"
            f"  ({result.see_rate_per_day * 365.25:.1f} events/year)\n\n"
            f"Dose Breakdown (mean):\n"
            f"  GCR:     {float(np.mean(result.gcr_dose_krad)):8.2f} krad\n"
            f"  Trapped: {float(np.mean(result.trapped_dose_krad)):8.2f} krad\n"
            f"  SPE:     {float(np.mean(result.spe_dose_krad)):8.2f} krad"
        )
        ax4.text(
            0.1,
            0.9,
            summary,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor=margin_color, alpha=0.8),
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Radiation analysis plot saved to {output_path}")

        return fig
