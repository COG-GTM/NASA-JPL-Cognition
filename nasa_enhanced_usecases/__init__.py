"""
NASA Enhanced Use Cases Module

Real-world use cases pulled from NASA/NASA-JPL open-source repositories,
enhanced and accelerated with AI-assisted development. Each sub-module
demonstrates how Devin can take existing manual workflows and dramatically
improve them.

Source repositories:
- nasa/progpy: Prognostic Python Packages for remaining useful life estimation
- nasa-jpl/captoolkit: Cryosphere altimetry processing
- nasa-jpl/FlightView: Real-time imaging spectroscopy tools
- nasa/K2CE: Kepler K2 Cadence Events for light curve cleaning
"""

from nasa_enhanced_usecases.cryosphere_analysis.ice_sheet_analysis import IceSheetAnalyzer
from nasa_enhanced_usecases.prognostics.battery_rul import BatteryRULPredictor
from nasa_enhanced_usecases.spectral_processing.hyperspectral_analysis import (
    HyperspectralAnalyzer,
)

__all__ = [
    "BatteryRULPredictor",
    "IceSheetAnalyzer",
    "HyperspectralAnalyzer",
]
