"""
Science Data Processing & Visualization Module

Inspired by NASA-JPL open-source tools:
- nasa-jpl/captoolkit: Cryosphere Altimetry Processing Toolkit
- nasa-jpl/autoRIFT: Pixel displacement algorithms for ice velocity
- nasa-jpl/its_live: Global glacier velocity datasets
- nasa/HLS-Data-Resources: Harmonized Landsat Sentinel-2 data

This module demonstrates how AI-assisted development can accelerate
science data pipelines — from raw data ingestion to publication-ready
visualizations — tasks that traditionally require weeks of manual coding.
"""

from science_data_processing.asteroid_tracking import NearEarthObjectTracker
from science_data_processing.exoplanet_analysis import ExoplanetAnalyzer
from science_data_processing.mars_surface_analysis import MarsSurfaceAnalyzer

__all__ = [
    "MarsSurfaceAnalyzer",
    "NearEarthObjectTracker",
    "ExoplanetAnalyzer",
]
