"""
Monte Carlo Simulation Module

Inspired by NASA open-source tools:
- nasa/SMCPy: Sequential Monte Carlo with Python (126 stars)
- nasa/MCMCPy: Markov Chain Monte Carlo sampler
- nasa-jpl/MonteCop: Spacecraft trajectory solution transfer

This module implements Monte Carlo methods for spacecraft mission
analysis — trajectory uncertainty quantification, mission reliability
assessment, and radiation environment modeling — the kinds of analyses
that JPL runs thousands of times per mission design cycle.
"""

from monte_carlo_simulation.mission_reliability import MissionReliabilitySimulator
from monte_carlo_simulation.radiation_environment import RadiationEnvironmentModel
from monte_carlo_simulation.trajectory_uncertainty import TrajectoryUncertaintyAnalyzer

__all__ = [
    "TrajectoryUncertaintyAnalyzer",
    "MissionReliabilitySimulator",
    "RadiationEnvironmentModel",
]
