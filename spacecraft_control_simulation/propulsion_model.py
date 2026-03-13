"""
Spacecraft Propulsion System Modeling

Models chemical and electric propulsion systems for trade study analysis,
including thrust curves, specific impulse, and propellant mass estimation.

Relevant to JPL mission design where propulsion trades drive architecture
decisions — typically modeled in MATLAB/Simulink, here in pure Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

G0 = 9.80665  # Standard gravity (m/s^2)


class PropulsionType(Enum):
    """Types of spacecraft propulsion systems."""

    CHEMICAL_BIPROPELLANT = "chemical_biprop"
    CHEMICAL_MONOPROPELLANT = "chemical_monoprop"
    COLD_GAS = "cold_gas"
    ION_ELECTRIC = "ion_electric"
    HALL_EFFECT = "hall_effect"
    SOLAR_SAIL = "solar_sail"


@dataclass
class PropulsionConfig:
    """Configuration for a propulsion system."""

    name: str
    propulsion_type: PropulsionType
    specific_impulse_s: float  # Isp in seconds
    thrust_n: float  # Thrust in Newtons
    mass_flow_rate_kgs: float  # kg/s
    dry_mass_kg: float  # Engine dry mass
    efficiency: float = 0.9
    max_burn_time_s: float = float("inf")


@dataclass
class PropellantBudget:
    """Propellant mass budget for a mission."""

    delta_v_ms: float
    spacecraft_dry_mass_kg: float
    propellant_mass_kg: float
    total_mass_kg: float
    mass_ratio: float
    burn_time_s: float
    propulsion_config: PropulsionConfig


@dataclass
class TradeStudyResult:
    """Result of propulsion trade study across multiple systems."""

    configs: list[PropulsionConfig]
    budgets: list[PropellantBudget]
    delta_v_ms: float
    spacecraft_dry_mass_kg: float
    optimal_config: PropulsionConfig
    optimal_budget: PropellantBudget


# Standard propulsion configurations based on real spacecraft systems
STANDARD_CONFIGS = {
    "biprop_main": PropulsionConfig(
        name="Bipropellant Main Engine (like JPL's MRO)",
        propulsion_type=PropulsionType.CHEMICAL_BIPROPELLANT,
        specific_impulse_s=320.0,
        thrust_n=445.0,
        mass_flow_rate_kgs=0.142,
        dry_mass_kg=5.0,
    ),
    "monoprop_rcs": PropulsionConfig(
        name="Hydrazine Monoprop (like Cassini RCS)",
        propulsion_type=PropulsionType.CHEMICAL_MONOPROPELLANT,
        specific_impulse_s=230.0,
        thrust_n=22.0,
        mass_flow_rate_kgs=0.0098,
        dry_mass_kg=1.5,
    ),
    "cold_gas": PropulsionConfig(
        name="Cold Gas N2 Thruster",
        propulsion_type=PropulsionType.COLD_GAS,
        specific_impulse_s=65.0,
        thrust_n=0.5,
        mass_flow_rate_kgs=7.84e-4,
        dry_mass_kg=0.3,
    ),
    "ion_nstar": PropulsionConfig(
        name="NSTAR Ion Engine (Dawn spacecraft)",
        propulsion_type=PropulsionType.ION_ELECTRIC,
        specific_impulse_s=3100.0,
        thrust_n=0.092,
        mass_flow_rate_kgs=3.02e-6,
        dry_mass_kg=8.2,
        max_burn_time_s=30000 * 3600,  # ~30,000 hours rated life
    ),
    "hall_spt100": PropulsionConfig(
        name="SPT-100 Hall Thruster",
        propulsion_type=PropulsionType.HALL_EFFECT,
        specific_impulse_s=1600.0,
        thrust_n=0.083,
        mass_flow_rate_kgs=5.29e-6,
        dry_mass_kg=3.5,
    ),
}


class PropulsionSystemModel:
    """
    Spacecraft propulsion system modeling for trade studies.

    Provides:
    - Tsiolkovsky rocket equation calculations
    - Propellant mass budgeting
    - Multi-system trade studies
    - Thrust curve and performance visualization

    Example:
        >>> model = PropulsionSystemModel()
        >>> budget = model.compute_propellant_budget(
        ...     delta_v_ms=2000.0,
        ...     spacecraft_dry_mass_kg=500.0,
        ...     config=STANDARD_CONFIGS["biprop_main"]
        ... )
        >>> print(f"Propellant needed: {budget.propellant_mass_kg:.1f} kg")
        >>> trade = model.run_trade_study(delta_v_ms=2000.0, dry_mass_kg=500.0)
        >>> model.plot_trade_study(trade, output_path="propulsion_trade.png")
    """

    def compute_propellant_budget(
        self,
        delta_v_ms: float,
        spacecraft_dry_mass_kg: float,
        config: PropulsionConfig,
    ) -> PropellantBudget:
        """
        Compute propellant mass budget using the Tsiolkovsky rocket equation.

        m_propellant = m_dry * (exp(dV / (Isp * g0)) - 1)

        Args:
            delta_v_ms: Required delta-V in m/s
            spacecraft_dry_mass_kg: Spacecraft dry mass in kg
            config: Propulsion system configuration

        Returns:
            PropellantBudget with mass and timing estimates
        """
        ve = config.specific_impulse_s * G0  # Exhaust velocity
        mass_ratio = np.exp(delta_v_ms / ve)
        total_mass = spacecraft_dry_mass_kg * mass_ratio
        propellant_mass = total_mass - spacecraft_dry_mass_kg

        # Burn time estimate
        if config.mass_flow_rate_kgs > 0:
            burn_time = propellant_mass / config.mass_flow_rate_kgs
        else:
            burn_time = float("inf")

        return PropellantBudget(
            delta_v_ms=delta_v_ms,
            spacecraft_dry_mass_kg=spacecraft_dry_mass_kg,
            propellant_mass_kg=float(propellant_mass),
            total_mass_kg=float(total_mass),
            mass_ratio=float(mass_ratio),
            burn_time_s=float(burn_time),
            propulsion_config=config,
        )

    def run_trade_study(
        self,
        delta_v_ms: float,
        dry_mass_kg: float,
        configs: Optional[dict[str, PropulsionConfig]] = None,
    ) -> TradeStudyResult:
        """
        Run a propulsion trade study across multiple systems.

        Args:
            delta_v_ms: Mission delta-V requirement (m/s)
            dry_mass_kg: Spacecraft dry mass (kg)
            configs: Dict of propulsion configs (defaults to STANDARD_CONFIGS)

        Returns:
            TradeStudyResult with all budgets and optimal selection
        """
        if configs is None:
            configs = STANDARD_CONFIGS

        config_list = list(configs.values())
        budgets = [
            self.compute_propellant_budget(delta_v_ms, dry_mass_kg, cfg) for cfg in config_list
        ]

        # Optimal = minimum total mass (propellant + engine)
        total_system_masses = [
            b.propellant_mass_kg + b.propulsion_config.dry_mass_kg for b in budgets
        ]
        optimal_idx = int(np.argmin(total_system_masses))

        logger.info(
            f"Trade study complete: dV={delta_v_ms}m/s, Optimal={config_list[optimal_idx].name}"
        )

        return TradeStudyResult(
            configs=config_list,
            budgets=budgets,
            delta_v_ms=delta_v_ms,
            spacecraft_dry_mass_kg=dry_mass_kg,
            optimal_config=config_list[optimal_idx],
            optimal_budget=budgets[optimal_idx],
        )

    def plot_trade_study(
        self,
        result: TradeStudyResult,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate propulsion trade study visualization.

        Args:
            result: TradeStudyResult from run_trade_study
            output_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Propulsion System Trade Study\n"
            f"(ΔV = {result.delta_v_ms:.0f} m/s, "
            f"Dry Mass = {result.spacecraft_dry_mass_kg:.0f} kg)",
            fontsize=16,
            fontweight="bold",
        )

        names = [c.name.split("(")[0].strip() for c in result.configs]
        prop_masses = [b.propellant_mass_kg for b in result.budgets]
        isps = [c.specific_impulse_s for c in result.configs]
        thrusts = [c.thrust_n for c in result.configs]

        # Panel 1: Propellant mass comparison
        ax1 = axes[0, 0]
        colors = [
            "#2ecc71" if i == result.configs.index(result.optimal_config) else "#3498db"
            for i in range(len(names))
        ]
        bars = ax1.barh(names, prop_masses, color=colors, edgecolor="black")
        ax1.set_xlabel("Propellant Mass (kg)")
        ax1.set_title("Propellant Mass Comparison")
        ax1.grid(True, alpha=0.3, axis="x")
        for bar, mass in zip(bars, prop_masses):
            ax1.text(
                bar.get_width() + max(prop_masses) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{mass:.1f} kg",
                va="center",
                fontsize=9,
            )

        # Panel 2: Isp vs Thrust scatter
        ax2 = axes[0, 1]
        ax2.scatter(isps, thrusts, s=200, c=prop_masses, cmap="RdYlGn_r", edgecolors="black")
        for i, name in enumerate(names):
            ax2.annotate(
                name[:15],
                (isps[i], thrusts[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax2.set_xlabel("Specific Impulse (s)")
        ax2.set_ylabel("Thrust (N)")
        ax2.set_title("Isp vs Thrust Trade Space")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Delta-V capability curves
        ax3 = axes[1, 0]
        dv_range = np.linspace(100, 5000, 200)
        for config in result.configs:
            ve = config.specific_impulse_s * G0
            prop_mass_curve = result.spacecraft_dry_mass_kg * (np.exp(dv_range / ve) - 1)
            ax3.plot(dv_range, prop_mass_curve, linewidth=1.5, label=config.name[:20])
        ax3.axvline(
            x=result.delta_v_ms,
            color="red",
            linestyle="--",
            label=f"Mission ΔV ({result.delta_v_ms:.0f} m/s)",
        )
        ax3.set_xlabel("Delta-V (m/s)")
        ax3.set_ylabel("Propellant Mass (kg)")
        ax3.set_title("Delta-V Capability Curves")
        ax3.legend(fontsize=7, loc="upper left")
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

        # Panel 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis("off")
        opt = result.optimal_budget
        summary = (
            f"OPTIMAL PROPULSION SYSTEM\n"
            f"{'=' * 40}\n\n"
            f"System: {result.optimal_config.name}\n"
            f"Type: {result.optimal_config.propulsion_type.value}\n"
            f"Isp: {result.optimal_config.specific_impulse_s:.0f} s\n"
            f"Thrust: {result.optimal_config.thrust_n:.3f} N\n\n"
            f"Mission Parameters:\n"
            f"  Delta-V: {opt.delta_v_ms:.0f} m/s\n"
            f"  Dry Mass: {opt.spacecraft_dry_mass_kg:.0f} kg\n"
            f"  Propellant: {opt.propellant_mass_kg:.1f} kg\n"
            f"  Total Mass: {opt.total_mass_kg:.1f} kg\n"
            f"  Mass Ratio: {opt.mass_ratio:.3f}\n"
            f"  Burn Time: {opt.burn_time_s / 3600:.1f} hr"
        )
        ax4.text(
            0.1,
            0.9,
            summary,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#e8f8e8", alpha=0.8),
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Trade study plot saved to {output_path}")

        return fig
