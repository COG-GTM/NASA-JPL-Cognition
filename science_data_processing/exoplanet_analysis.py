"""
Exoplanet Data Analysis & Habitability Assessment

Processes exoplanet catalog data from NASA's Exoplanet Archive to
identify habitable zone candidates and generate science visualizations.

Inspired by workflows in:
- nasa/Kepler-PyKE: Kepler light curve analysis
- nasa/K2CE: K2 Cadence Events processing

Data source: NASA Exoplanet Archive (Caltech/IPAC, managed for NASA)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

EXOPLANET_ARCHIVE_API = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


@dataclass
class Exoplanet:
    """Represents an exoplanet with key physical and orbital parameters."""

    name: str
    host_star: str
    discovery_method: str
    orbital_period_days: float
    semi_major_axis_au: float
    planet_radius_earth: float
    planet_mass_earth: float
    equilibrium_temp_k: float
    stellar_luminosity: float
    stellar_temp_k: float
    discovery_year: int
    habitable_zone: bool = False


@dataclass
class HabitabilityScore:
    """Habitability assessment for an exoplanet."""

    planet: Exoplanet
    earth_similarity_index: float
    hz_inner_au: float
    hz_outer_au: float
    in_habitable_zone: bool
    size_score: float
    temp_score: float
    overall_score: float
    classification: str


class ExoplanetAnalyzer:
    """
    Exoplanet catalog analysis with habitability assessment.

    Fetches confirmed exoplanet data from NASA's Exoplanet Archive,
    computes Earth Similarity Index (ESI) and habitable zone boundaries,
    and generates publication-quality visualizations.

    Example:
        >>> analyzer = ExoplanetAnalyzer()
        >>> planets = analyzer.fetch_confirmed_exoplanets(limit=500)
        >>> candidates = analyzer.find_habitable_candidates(planets)
        >>> print(f"Found {len(candidates)} habitable zone candidates")
        >>> analyzer.generate_habitability_report(planets, output_path="exoplanet_report.png")
    """

    def __init__(self):
        self.session = requests.Session()

    def fetch_confirmed_exoplanets(self, limit: int = 500) -> list[Exoplanet]:
        """
        Fetch confirmed exoplanet data from NASA Exoplanet Archive.

        Args:
            limit: Maximum number of results

        Returns:
            List of Exoplanet objects with physical/orbital parameters
        """
        query = (
            "SELECT pl_name, hostname, discoverymethod, pl_orbper, pl_orbsmax, "
            "pl_rade, pl_bmasse, pl_eqt, st_lum, st_teff, disc_year "
            "FROM ps WHERE pl_rade IS NOT NULL AND pl_orbsmax IS NOT NULL "
            "AND default_flag = 1 ORDER BY disc_year DESC "
        )

        params = {
            "query": query,
            "format": "json",
        }

        logger.info(f"Fetching up to {limit} confirmed exoplanets from NASA Archive")

        try:
            response = self.session.get(EXOPLANET_ARCHIVE_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Failed to fetch from Exoplanet Archive: {e}")
            logger.info("Using synthetic exoplanet data")
            return self._generate_synthetic_exoplanets(count=limit)

        planets = []
        for entry in data[:limit]:
            try:
                planet = Exoplanet(
                    name=entry.get("pl_name", "Unknown"),
                    host_star=entry.get("hostname", "Unknown"),
                    discovery_method=entry.get("discoverymethod", "Unknown"),
                    orbital_period_days=float(entry.get("pl_orbper", 0) or 0),
                    semi_major_axis_au=float(entry.get("pl_orbsmax", 0) or 0),
                    planet_radius_earth=float(entry.get("pl_rade", 0) or 0),
                    planet_mass_earth=float(entry.get("pl_bmasse", 0) or 0),
                    equilibrium_temp_k=float(entry.get("pl_eqt", 0) or 0),
                    stellar_luminosity=float(entry.get("st_lum", 0) or 0),
                    stellar_temp_k=float(entry.get("st_teff", 5778) or 5778),
                    discovery_year=int(entry.get("disc_year", 2020) or 2020),
                )
                planets.append(planet)
            except (TypeError, ValueError):
                continue

        logger.info(f"Retrieved {len(planets)} confirmed exoplanets")
        return planets

    def compute_habitable_zone(self, stellar_luminosity: float) -> tuple[float, float]:
        """
        Compute conservative habitable zone boundaries.

        Uses the Kopparapu et al. (2013) model for habitable zone
        inner and outer edges based on stellar luminosity.

        Args:
            stellar_luminosity: Stellar luminosity in solar luminosities (log10)

        Returns:
            Tuple of (inner_edge_au, outer_edge_au)
        """
        luminosity_solar = 10**stellar_luminosity if stellar_luminosity != 0 else 1.0
        hz_inner = 0.95 * np.sqrt(luminosity_solar)
        hz_outer = 1.67 * np.sqrt(luminosity_solar)
        return float(hz_inner), float(hz_outer)

    def compute_earth_similarity_index(self, planet: Exoplanet) -> float:
        """
        Compute Earth Similarity Index (ESI) for an exoplanet.

        ESI ranges from 0 (completely unlike Earth) to 1 (Earth twin).
        Based on Schulze-Makuch et al. (2011).

        Args:
            planet: Exoplanet with physical parameters

        Returns:
            ESI value between 0 and 1
        """
        earth_radius = 1.0
        earth_temp = 255.0  # Earth equilibrium temperature in K

        if planet.planet_radius_earth <= 0 or planet.equilibrium_temp_k <= 0:
            return 0.0

        # Radius similarity
        radius_ratio = planet.planet_radius_earth / earth_radius
        esi_radius = 1.0 - abs((radius_ratio - 1.0) / (radius_ratio + 1.0))

        # Temperature similarity
        temp = planet.equilibrium_temp_k
        temp_ratio = temp / earth_temp
        esi_temp = 1.0 - abs((temp_ratio - 1.0) / (temp_ratio + 1.0))

        # Combined ESI (geometric mean of component similarities)
        esi = float((esi_radius**0.57) * (esi_temp**0.43))
        return max(0.0, min(1.0, esi))

    def find_habitable_candidates(
        self, planets: list[Exoplanet], esi_threshold: float = 0.6
    ) -> list[HabitabilityScore]:
        """
        Screen exoplanets for habitability potential.

        Args:
            planets: List of Exoplanet objects
            esi_threshold: Minimum ESI score to be considered a candidate

        Returns:
            List of HabitabilityScore for candidates above threshold
        """
        candidates = []
        for planet in planets:
            hz_inner, hz_outer = self.compute_habitable_zone(planet.stellar_luminosity)
            in_hz = hz_inner <= planet.semi_major_axis_au <= hz_outer

            esi = self.compute_earth_similarity_index(planet)

            # Size score: peaks at 1 Earth radius
            if planet.planet_radius_earth > 0:
                size_score = float(np.exp(-0.5 * ((planet.planet_radius_earth - 1.0) / 0.5) ** 2))
            else:
                size_score = 0.0

            # Temperature score: peaks at 255 K
            if planet.equilibrium_temp_k > 0:
                temp_score = float(np.exp(-0.5 * ((planet.equilibrium_temp_k - 255) / 50) ** 2))
            else:
                temp_score = 0.0

            overall = 0.4 * esi + 0.3 * float(in_hz) + 0.2 * size_score + 0.1 * temp_score

            if overall >= esi_threshold * 0.5:
                if overall > 0.7:
                    classification = "Prime Candidate"
                elif overall > 0.5:
                    classification = "Promising"
                elif overall > 0.3:
                    classification = "Worth Investigating"
                else:
                    classification = "Unlikely Habitable"

                candidates.append(
                    HabitabilityScore(
                        planet=planet,
                        earth_similarity_index=esi,
                        hz_inner_au=hz_inner,
                        hz_outer_au=hz_outer,
                        in_habitable_zone=in_hz,
                        size_score=size_score,
                        temp_score=temp_score,
                        overall_score=overall,
                        classification=classification,
                    )
                )

        candidates.sort(key=lambda c: c.overall_score, reverse=True)
        return candidates

    def generate_habitability_report(
        self,
        planets: list[Exoplanet],
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate a comprehensive exoplanet habitability visualization.

        Creates a multi-panel figure with:
        - Habitable zone diagram
        - ESI distribution
        - Discovery method breakdown
        - Mass-radius relationship

        Args:
            planets: List of Exoplanet objects
            output_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Exoplanet Habitability Analysis\n(Data: NASA Exoplanet Archive)",
            fontsize=16,
            fontweight="bold",
        )

        valid_planets = [p for p in planets if p.semi_major_axis_au > 0]

        # Panel 1: Habitable Zone diagram
        ax1 = axes[0, 0]
        for planet in valid_planets:
            hz_inner, hz_outer = self.compute_habitable_zone(planet.stellar_luminosity)
            in_hz = hz_inner <= planet.semi_major_axis_au <= hz_outer
            color = "green" if in_hz else "gray"
            alpha = 0.8 if in_hz else 0.2
            size = max(5, min(200, planet.planet_radius_earth * 20))
            ax1.scatter(
                planet.semi_major_axis_au,
                planet.stellar_temp_k,
                s=size,
                c=color,
                alpha=alpha,
                edgecolors="black",
                linewidth=0.5,
            )

        # Add Earth reference
        ax1.scatter(1.0, 5778, s=100, c="blue", marker="*", zorder=5, label="Earth")
        ax1.set_xlabel("Semi-Major Axis (AU)")
        ax1.set_ylabel("Stellar Temperature (K)")
        ax1.set_title("Habitable Zone Diagram")
        ax1.set_xscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: ESI distribution
        ax2 = axes[0, 1]
        esi_values = [
            self.compute_earth_similarity_index(p)
            for p in valid_planets
            if p.equilibrium_temp_k > 0
        ]
        if esi_values:
            ax2.hist(esi_values, bins=30, color="teal", edgecolor="black", alpha=0.7)
            ax2.axvline(x=0.8, color="green", linestyle="--", label="Earth-like threshold")
        ax2.set_xlabel("Earth Similarity Index (ESI)")
        ax2.set_ylabel("Count")
        ax2.set_title("ESI Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Discovery methods
        ax3 = axes[1, 0]
        methods = pd.Series([p.discovery_method for p in planets])
        method_counts = methods.value_counts().head(8)
        method_counts.plot.barh(ax=ax3, color="steelblue", edgecolor="black")
        ax3.set_xlabel("Number of Planets")
        ax3.set_title("Discovery Methods")
        ax3.grid(True, alpha=0.3, axis="x")

        # Panel 4: Mass-Radius relationship
        ax4 = axes[1, 1]
        valid_mr = [p for p in planets if p.planet_mass_earth > 0 and p.planet_radius_earth > 0]
        if valid_mr:
            masses = [p.planet_mass_earth for p in valid_mr]
            radii = [p.planet_radius_earth for p in valid_mr]
            temps = [p.equilibrium_temp_k if p.equilibrium_temp_k > 0 else 300 for p in valid_mr]

            sc = ax4.scatter(masses, radii, c=temps, cmap="RdYlBu_r", s=20, alpha=0.6)
            fig.colorbar(sc, ax=ax4, label="Equilibrium Temp (K)")

            # Earth reference
            ax4.scatter(1.0, 1.0, s=100, c="blue", marker="*", zorder=5, label="Earth")

        ax4.set_xlabel("Mass (Earth masses)")
        ax4.set_ylabel("Radius (Earth radii)")
        ax4.set_title("Mass-Radius Relationship")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Exoplanet report saved to {output_path}")

        return fig

    def _generate_synthetic_exoplanets(self, count: int = 500) -> list[Exoplanet]:
        """Generate synthetic exoplanet data for offline demonstration."""
        rng = np.random.default_rng(42)
        methods = ["Transit", "Radial Velocity", "Direct Imaging", "Microlensing", "Transit Timing"]
        method_weights = [0.75, 0.15, 0.04, 0.03, 0.03]

        planets = []
        for i in range(count):
            radius = float(rng.lognormal(0.5, 1.0))
            radius = min(radius, 25.0)
            mass = float(radius**2.06 * rng.uniform(0.5, 2.0))
            semi_major = float(rng.lognormal(-0.5, 1.5))
            stellar_lum = float(rng.normal(0, 0.5))
            stellar_temp = float(rng.normal(5500, 1000))
            stellar_temp = max(2500, min(10000, stellar_temp))

            luminosity_solar = 10**stellar_lum
            eq_temp = (
                float(278 * (luminosity_solar**0.25) / (semi_major**0.5)) if semi_major > 0 else 300
            )

            method_idx = rng.choice(len(methods), p=method_weights)

            planets.append(
                Exoplanet(
                    name=f"Synthetic-{i + 1:04d}b",
                    host_star=f"Star-{i + 1:04d}",
                    discovery_method=methods[method_idx],
                    orbital_period_days=float(semi_major**1.5 * 365.25),
                    semi_major_axis_au=semi_major,
                    planet_radius_earth=radius,
                    planet_mass_earth=mass,
                    equilibrium_temp_k=eq_temp,
                    stellar_luminosity=stellar_lum,
                    stellar_temp_k=stellar_temp,
                    discovery_year=int(rng.integers(1995, 2025)),
                )
            )

        return planets
