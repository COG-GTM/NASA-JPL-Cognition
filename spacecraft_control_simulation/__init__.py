"""
Spacecraft Control Simulation Module

Inspired by NASA-JPL open-source tools:
- nasa-jpl/SAAS: Simulation for the Analysis of Autonomy at the System Level
- nasa-jpl/MonteCop: Spacecraft trajectory solution transfer tools
- nasa-jpl/lowfssim: Roman-CGI optical model of LOWFS

This module provides Python-based spacecraft dynamics and control
simulations — the open-source equivalent of Simulink models — covering
attitude control, orbital mechanics, and propulsion system modeling.
"""

from spacecraft_control_simulation.attitude_control import SpacecraftAttitudeController
from spacecraft_control_simulation.orbital_mechanics import OrbitalMechanicsSimulator
from spacecraft_control_simulation.propulsion_model import PropulsionSystemModel

__all__ = [
    "SpacecraftAttitudeController",
    "OrbitalMechanicsSimulator",
    "PropulsionSystemModel",
]
