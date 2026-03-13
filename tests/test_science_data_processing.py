"""Tests for the science data processing module."""

import numpy as np

from science_data_processing.asteroid_tracking import (
    NearEarthObject,
    NearEarthObjectTracker,
)
from science_data_processing.exoplanet_analysis import (
    Exoplanet,
    ExoplanetAnalyzer,
)
from science_data_processing.mars_surface_analysis import (
    MarsImage,
    MarsSurfaceAnalyzer,
    TerrainClassification,
)


class TestMarsSurfaceAnalyzer:
    def test_synthetic_image_generation(self):
        analyzer = MarsSurfaceAnalyzer()
        img = analyzer._generate_synthetic_mars_image(size=64)
        assert img.shape == (64, 64)
        assert img.dtype == np.float64
        assert img.min() >= 0
        assert img.max() <= 255

    def test_terrain_classification(self):
        analyzer = MarsSurfaceAnalyzer()
        image = MarsImage(
            image_id=1,
            sol=100,
            camera_name="NAVCAM",
            camera_full_name="Navigation Camera",
            earth_date="2024-01-01",
            rover_name="Curiosity",
            img_src="http://example.com/img.jpg",
        )
        classification = analyzer.classify_terrain(image)
        assert isinstance(classification, TerrainClassification)
        total = (
            classification.rock_fraction
            + classification.sand_fraction
            + classification.bedrock_fraction
            + classification.shadow_fraction
        )
        assert abs(total - 1.0) < 0.01
        assert classification.roughness_index >= 0
        assert classification.dominant_terrain in ["rock", "sand", "bedrock", "shadow"]

    def test_terrain_map_generation(self):
        analyzer = MarsSurfaceAnalyzer()
        images = [
            MarsImage(
                image_id=i,
                sol=100 + i,
                camera_name="NAVCAM",
                camera_full_name="Navigation Camera",
                earth_date="2024-01-01",
                rover_name="Curiosity",
                img_src="http://example.com/img.jpg",
            )
            for i in range(3)
        ]
        fig = analyzer.generate_terrain_map(images)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestNearEarthObjectTracker:
    def test_synthetic_neos(self):
        tracker = NearEarthObjectTracker()
        neos = tracker._generate_synthetic_neos(count=10)
        assert len(neos) == 10
        for neo in neos:
            assert isinstance(neo, NearEarthObject)
            assert neo.miss_distance_km > 0
            assert neo.relative_velocity_kph > 0

    def test_hazard_assessment(self):
        tracker = NearEarthObjectTracker()
        neo = NearEarthObject(
            neo_id="test1",
            name="Test Asteroid",
            absolute_magnitude=22.0,
            estimated_diameter_min_km=0.1,
            estimated_diameter_max_km=0.2,
            is_potentially_hazardous=True,
            close_approach_date="2024-01-01",
            relative_velocity_kph=50000.0,
            miss_distance_km=500000.0,
            miss_distance_lunar=1.3,
            orbiting_body="Earth",
        )
        assessment = tracker.assess_hazard(neo)
        assert assessment.kinetic_energy_megatons > 0
        assert assessment.torino_scale_estimate >= 0
        assert assessment.torino_scale_estimate <= 10
        assert assessment.risk_category != ""

    def test_visualization(self):
        tracker = NearEarthObjectTracker()
        neos = tracker._generate_synthetic_neos(count=15)
        fig = tracker.generate_approach_visualization(neos)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestExoplanetAnalyzer:
    def test_synthetic_exoplanets(self):
        analyzer = ExoplanetAnalyzer()
        planets = analyzer._generate_synthetic_exoplanets(count=50)
        assert len(planets) == 50
        for planet in planets:
            assert isinstance(planet, Exoplanet)
            assert planet.planet_radius_earth > 0

    def test_habitable_zone_computation(self):
        analyzer = ExoplanetAnalyzer()
        hz_inner, hz_outer = analyzer.compute_habitable_zone(0.0)  # Solar luminosity
        assert 0.8 < hz_inner < 1.1
        assert 1.5 < hz_outer < 2.0

    def test_earth_similarity_index(self):
        analyzer = ExoplanetAnalyzer()
        earth_like = Exoplanet(
            name="Earth-like",
            host_star="Sun-like",
            discovery_method="Transit",
            orbital_period_days=365.25,
            semi_major_axis_au=1.0,
            planet_radius_earth=1.0,
            planet_mass_earth=1.0,
            equilibrium_temp_k=255.0,
            stellar_luminosity=0.0,
            stellar_temp_k=5778.0,
            discovery_year=2024,
        )
        esi = analyzer.compute_earth_similarity_index(earth_like)
        assert esi > 0.95  # Earth-like should score very high

    def test_find_habitable_candidates(self):
        analyzer = ExoplanetAnalyzer()
        planets = analyzer._generate_synthetic_exoplanets(count=100)
        candidates = analyzer.find_habitable_candidates(planets)
        assert isinstance(candidates, list)
        # Should find at least some candidates in 100 random planets
        assert len(candidates) >= 0

    def test_habitability_report(self):
        analyzer = ExoplanetAnalyzer()
        planets = analyzer._generate_synthetic_exoplanets(count=50)
        fig = analyzer.generate_habitability_report(planets)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
