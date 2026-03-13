"""
Mission Reliability Monte Carlo Simulation

Models spacecraft subsystem reliability using Monte Carlo methods to
estimate mission success probability — a critical JPL Systems Engineering
task for every mission from proposal through operations.

Models failure modes including:
- Component wear-out (Weibull distribution)
- Random failures (Exponential distribution)
- Radiation-induced single-event effects
- Redundancy switching
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of component failure modes."""

    RANDOM = "random"  # Exponential (constant hazard rate)
    WEAROUT = "wearout"  # Weibull (increasing hazard rate)
    RADIATION_SEE = "radiation_see"  # Single Event Effects
    THERMAL_CYCLING = "thermal_cycling"


@dataclass
class SubsystemConfig:
    """Configuration for a spacecraft subsystem reliability model."""

    name: str
    failure_mode: FailureMode
    mtbf_hours: float  # Mean Time Between Failures
    weibull_shape: float = 1.0  # Shape parameter (beta); 1.0 = exponential
    redundancy_level: int = 1  # 1 = no redundancy, 2 = single redundant, etc.
    is_critical: bool = True  # Mission-critical subsystem
    repair_possible: bool = False


@dataclass
class MissionConfig:
    """Mission-level configuration."""

    name: str
    duration_hours: float
    subsystems: list[SubsystemConfig] = field(default_factory=list)
    mission_success_requires_all_critical: bool = True


@dataclass
class ReliabilityResult:
    """Results from mission reliability Monte Carlo."""

    n_samples: int
    mission_config: MissionConfig
    mission_success_rate: float
    subsystem_survival_rates: dict[str, float]
    first_failure_times: dict[str, np.ndarray]
    mission_lifetime_distribution: np.ndarray
    mean_time_to_first_failure_hours: float
    percentile_10_lifetime_hours: float
    availability: float


# Typical deep-space mission subsystem configurations
DEEP_SPACE_SUBSYSTEMS = [
    SubsystemConfig(
        name="Command & Data Handling",
        failure_mode=FailureMode.RANDOM,
        mtbf_hours=500_000,
        redundancy_level=2,
        is_critical=True,
    ),
    SubsystemConfig(
        name="Power (Solar Array)",
        failure_mode=FailureMode.WEAROUT,
        mtbf_hours=200_000,
        weibull_shape=2.5,
        redundancy_level=1,
        is_critical=True,
    ),
    SubsystemConfig(
        name="Telecommunications",
        failure_mode=FailureMode.RANDOM,
        mtbf_hours=300_000,
        redundancy_level=2,
        is_critical=True,
    ),
    SubsystemConfig(
        name="Attitude Control",
        failure_mode=FailureMode.RANDOM,
        mtbf_hours=400_000,
        redundancy_level=2,
        is_critical=True,
    ),
    SubsystemConfig(
        name="Propulsion",
        failure_mode=FailureMode.WEAROUT,
        mtbf_hours=150_000,
        weibull_shape=1.8,
        redundancy_level=1,
        is_critical=True,
    ),
    SubsystemConfig(
        name="Thermal Control",
        failure_mode=FailureMode.THERMAL_CYCLING,
        mtbf_hours=600_000,
        weibull_shape=3.0,
        redundancy_level=1,
        is_critical=False,
    ),
    SubsystemConfig(
        name="Science Instrument A",
        failure_mode=FailureMode.RADIATION_SEE,
        mtbf_hours=250_000,
        redundancy_level=1,
        is_critical=False,
    ),
    SubsystemConfig(
        name="Science Instrument B",
        failure_mode=FailureMode.RANDOM,
        mtbf_hours=350_000,
        redundancy_level=1,
        is_critical=False,
    ),
]


class MissionReliabilitySimulator:
    """
    Mission reliability analysis via Monte Carlo simulation.

    Simulates component-level failures across all spacecraft subsystems
    to estimate mission success probability — the same methodology used
    in JPL's mission design process (per NASA-STD-8729.1).

    Example:
        >>> sim = MissionReliabilitySimulator(n_samples=10000)
        >>> config = MissionConfig(
        ...     name="Europa Clipper",
        ...     duration_hours=6*365.25*24,  # 6 year mission
        ...     subsystems=DEEP_SPACE_SUBSYSTEMS,
        ... )
        >>> result = sim.run_reliability_analysis(config)
        >>> print(f"Mission success probability: {result.mission_success_rate:.3f}")
        >>> sim.plot_results(result, output_path="reliability.png")
    """

    def __init__(self, n_samples: int = 10000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def run_reliability_analysis(
        self,
        config: Optional[MissionConfig] = None,
    ) -> ReliabilityResult:
        """
        Run full mission reliability Monte Carlo analysis.

        For each sample, simulates failure times for every subsystem
        component and determines if the mission succeeds based on
        criticality and redundancy.

        Args:
            config: Mission configuration (defaults to Europa Clipper-like)

        Returns:
            ReliabilityResult with comprehensive statistics
        """
        if config is None:
            config = MissionConfig(
                name="Deep Space Mission",
                duration_hours=5 * 365.25 * 24,  # 5-year mission
                subsystems=DEEP_SPACE_SUBSYSTEMS,
            )

        logger.info(
            f"Running reliability analysis for '{config.name}': "
            f"{self.n_samples} samples, {config.duration_hours / 8766:.1f} year mission"
        )

        mission_lifetimes = np.zeros(self.n_samples)
        subsystem_failures: dict[str, np.ndarray] = {
            sub.name: np.zeros(self.n_samples) for sub in config.subsystems
        }
        mission_successes = np.zeros(self.n_samples, dtype=bool)

        for i in range(self.n_samples):
            first_critical_failure = config.duration_hours
            mission_ok = True

            for subsystem in config.subsystems:
                # Generate failure time for each redundant unit
                failure_times = self._generate_failure_times(subsystem, subsystem.redundancy_level)

                # System failure = all redundant units failed
                system_failure_time = max(failure_times)
                subsystem_failures[subsystem.name][i] = system_failure_time

                # Check if subsystem fails before mission end
                if system_failure_time < config.duration_hours:
                    if subsystem.is_critical:
                        mission_ok = False
                        if system_failure_time < first_critical_failure:
                            first_critical_failure = system_failure_time

            mission_lifetimes[i] = first_critical_failure
            mission_successes[i] = mission_ok

        # Compute statistics
        success_rate = float(np.mean(mission_successes))
        survival_rates = {
            name: float(np.mean(times >= config.duration_hours))
            for name, times in subsystem_failures.items()
        }
        mean_ttff = float(np.mean(mission_lifetimes))
        p10_lifetime = float(np.percentile(mission_lifetimes, 10))

        # Availability (fraction of mission time with all critical systems operational)
        critical_failures = [
            subsystem_failures[sub.name] for sub in config.subsystems if sub.is_critical
        ]
        if critical_failures:
            min_critical_times = np.min(np.column_stack(critical_failures), axis=1)
            availability = float(
                np.mean(np.minimum(min_critical_times, config.duration_hours))
                / config.duration_hours
            )
        else:
            availability = 1.0

        logger.info(
            f"Reliability analysis complete: P(success) = {success_rate:.3f}, "
            f"MTTFF = {mean_ttff / 8766:.1f} years"
        )

        return ReliabilityResult(
            n_samples=self.n_samples,
            mission_config=config,
            mission_success_rate=success_rate,
            subsystem_survival_rates=survival_rates,
            first_failure_times=subsystem_failures,
            mission_lifetime_distribution=mission_lifetimes,
            mean_time_to_first_failure_hours=mean_ttff,
            percentile_10_lifetime_hours=p10_lifetime,
            availability=availability,
        )

    def _generate_failure_times(self, subsystem: SubsystemConfig, n_units: int) -> list[float]:
        """Generate failure times for redundant subsystem units."""
        failure_times = []
        for _ in range(n_units):
            if subsystem.failure_mode == FailureMode.RANDOM:
                # Exponential distribution (constant failure rate)
                t_fail = float(self.rng.exponential(subsystem.mtbf_hours))

            elif subsystem.failure_mode in (FailureMode.WEAROUT, FailureMode.THERMAL_CYCLING):
                # Weibull distribution (increasing failure rate)
                scale = subsystem.mtbf_hours / np.exp(
                    np.log(math.gamma(1 + 1 / subsystem.weibull_shape))
                )
                t_fail = float(self.rng.weibull(subsystem.weibull_shape) * scale)

            elif subsystem.failure_mode == FailureMode.RADIATION_SEE:
                # Poisson process for single-event effects
                see_rate = 1.0 / subsystem.mtbf_hours
                # Time to first critical SEE
                t_fail = float(self.rng.exponential(1.0 / see_rate))

            else:
                t_fail = float(self.rng.exponential(subsystem.mtbf_hours))

            failure_times.append(t_fail)

        return failure_times

    def plot_results(
        self,
        result: ReliabilityResult,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate mission reliability visualization.

        Args:
            result: ReliabilityResult from run_reliability_analysis
            output_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        config = result.mission_config
        fig.suptitle(
            f"Mission Reliability Analysis: {config.name}\n"
            f"({result.n_samples:,} Monte Carlo samples, "
            f"{config.duration_hours / 8766:.1f}-year mission)",
            fontsize=16,
            fontweight="bold",
        )

        # Panel 1: Subsystem survival rates
        ax1 = axes[0, 0]
        names = list(result.subsystem_survival_rates.keys())
        rates = list(result.subsystem_survival_rates.values())
        colors = ["#2ecc71" if r > 0.95 else "#f39c12" if r > 0.9 else "#e74c3c" for r in rates]

        bars = ax1.barh(names, rates, color=colors, edgecolor="black")
        ax1.axvline(x=0.95, color="green", linestyle="--", alpha=0.5, label="95% threshold")
        ax1.set_xlabel("Survival Probability")
        ax1.set_title("Subsystem Survival Rates")
        ax1.set_xlim(0, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="x")
        for bar, rate in zip(bars, rates):
            ax1.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.3f}",
                va="center",
                fontsize=9,
            )

        # Panel 2: Mission lifetime distribution
        ax2 = axes[0, 1]
        lifetimes_years = result.mission_lifetime_distribution / 8766
        ax2.hist(
            lifetimes_years,
            bins=50,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
            density=True,
        )
        mission_years = config.duration_hours / 8766
        ax2.axvline(
            x=mission_years,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Required: {mission_years:.1f} yr",
        )
        ax2.axvline(
            x=result.mean_time_to_first_failure_hours / 8766,
            color="green",
            linestyle="--",
            label=f"Mean: {result.mean_time_to_first_failure_hours / 8766:.1f} yr",
        )
        ax2.set_xlabel("Time to First Critical Failure (years)")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("Mission Lifetime Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Reliability over time (survival curve)
        ax3 = axes[1, 0]
        time_points = np.linspace(0, config.duration_hours * 1.5, 200)
        time_years = time_points / 8766

        for subsystem in config.subsystems:
            if subsystem.is_critical:
                failure_times = result.first_failure_times[subsystem.name]
                survival_curve = [float(np.mean(failure_times >= t)) for t in time_points]
                ax3.plot(time_years, survival_curve, linewidth=1.5, label=subsystem.name[:15])

        ax3.axvline(x=mission_years, color="black", linestyle=":", alpha=0.5)
        ax3.set_xlabel("Mission Time (years)")
        ax3.set_ylabel("Survival Probability")
        ax3.set_title("Critical Subsystem Reliability Curves")
        ax3.legend(fontsize=8, loc="lower left")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)

        # Panel 4: Summary
        ax4 = axes[1, 1]
        ax4.axis("off")
        summary = (
            f"MISSION RELIABILITY SUMMARY\n"
            f"{'=' * 40}\n\n"
            f"Mission: {config.name}\n"
            f"Duration: {mission_years:.1f} years\n"
            f"Subsystems: {len(config.subsystems)}\n"
            f"Critical: {sum(1 for s in config.subsystems if s.is_critical)}\n\n"
            f"Mission Success Rate: {result.mission_success_rate:.1%}\n"
            f"Mean Time to Failure: {result.mean_time_to_first_failure_hours / 8766:.1f} yr\n"
            f"10th Percentile Life: {result.percentile_10_lifetime_hours / 8766:.1f} yr\n"
            f"System Availability: {result.availability:.4f}\n\n"
            f"{'PASS' if result.mission_success_rate > 0.95 else 'REVIEW NEEDED'}: "
            f"{'Meets' if result.mission_success_rate > 0.95 else 'Below'} "
            f"95% reliability threshold"
        )
        color = "#e8f8e8" if result.mission_success_rate > 0.95 else "#f8e8e8"
        ax4.text(
            0.1,
            0.9,
            summary,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Reliability plot saved to {output_path}")

        return fig
