"""Tests for the NASA enhanced use cases module."""

from nasa_enhanced_usecases.cryosphere_analysis.ice_sheet_analysis import (
    ANTARCTICA,
    GREENLAND,
    IceSheetAnalyzer,
)
from nasa_enhanced_usecases.prognostics.battery_rul import (
    BatteryRULPredictor,
    BatteryState,
)
from nasa_enhanced_usecases.spectral_processing.hyperspectral_analysis import (
    HyperspectralAnalyzer,
)


class TestBatteryRULPredictor:
    def test_synthetic_data_generation(self):
        predictor = BatteryRULPredictor()
        data = predictor.generate_synthetic_battery_data(n_cycles=100)
        assert len(data) == 100
        assert isinstance(data[0], BatteryState)
        # Capacity should decrease over time
        assert data[0].capacity_ah > data[-1].capacity_ah

    def test_rul_prediction(self):
        predictor = BatteryRULPredictor()
        data = predictor.generate_synthetic_battery_data(n_cycles=300)
        prediction = predictor.predict_rul(data, eol_threshold_ah=1.4)
        assert prediction.predicted_rul_cycles > 0
        assert prediction.current_cycle == 299
        assert prediction.current_capacity_ah > 0
        assert prediction.rul_lower_bound < prediction.predicted_rul_cycles
        assert prediction.rul_upper_bound > prediction.predicted_rul_cycles

    def test_degradation_rate(self):
        predictor = BatteryRULPredictor()
        data = predictor.generate_synthetic_battery_data(n_cycles=200)
        prediction = predictor.predict_rul(data)
        assert prediction.degradation_rate_per_cycle > 0

    def test_plot_prediction(self):
        predictor = BatteryRULPredictor()
        data = predictor.generate_synthetic_battery_data(n_cycles=200)
        prediction = predictor.predict_rul(data)
        fig = predictor.plot_prediction(data, prediction)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestIceSheetAnalyzer:
    def test_greenland_analysis(self):
        analyzer = IceSheetAnalyzer()
        result = analyzer.analyze_mass_change(
            region=GREENLAND,
            start_year=2003.0,
            end_year=2013.0,
        )
        assert result.region.name == "Greenland Ice Sheet"
        assert result.mass_rate_gt_yr < 0  # Greenland is losing mass
        assert len(result.time_years) > 0
        assert len(result.mass_change_gt) == len(result.time_years)

    def test_antarctica_analysis(self):
        analyzer = IceSheetAnalyzer()
        result = analyzer.analyze_mass_change(
            region=ANTARCTICA,
            start_year=2003.0,
            end_year=2013.0,
        )
        assert result.region.name == "Antarctic Ice Sheet"
        assert result.mass_rate_gt_yr < 0

    def test_sea_level_contribution(self):
        analyzer = IceSheetAnalyzer()
        result = analyzer.analyze_mass_change(region=GREENLAND)
        assert len(result.sea_level_contribution_mm) == len(result.time_years)
        # Mass loss → sea level rise
        assert result.sea_level_contribution_mm[-1] > 0

    def test_elevation_change_map(self):
        analyzer = IceSheetAnalyzer()
        result = analyzer.analyze_mass_change(region=GREENLAND)
        assert result.elevation_change_map.shape == (100, 100)

    def test_plot_results(self):
        analyzer = IceSheetAnalyzer()
        result = analyzer.analyze_mass_change(
            region=GREENLAND,
            start_year=2003.0,
            end_year=2013.0,
        )
        fig = analyzer.plot_results(result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestHyperspectralAnalyzer:
    def test_synthetic_scene(self):
        analyzer = HyperspectralAnalyzer()
        cube = analyzer.generate_synthetic_scene(n_rows=50, n_cols=50, n_bands=100)
        assert cube.data.shape == (50, 50, 100)
        assert len(cube.wavelengths_um) == 100
        assert cube.n_rows == 50
        assert cube.n_cols == 50

    def test_analysis_pipeline(self):
        analyzer = HyperspectralAnalyzer()
        cube = analyzer.generate_synthetic_scene(n_rows=30, n_cols=30, n_bands=50)
        result = analyzer.analyze(cube, n_classes=4, n_pca_components=5)
        assert result.classification_map.shape == (30, 30)
        assert result.n_classes == 4
        assert len(result.class_names) == 4
        assert result.anomaly_map.shape == (30, 30)
        assert len(result.mean_spectrum) == 50

    def test_plot_results(self):
        analyzer = HyperspectralAnalyzer()
        cube = analyzer.generate_synthetic_scene(n_rows=30, n_cols=30, n_bands=50)
        result = analyzer.analyze(cube, n_classes=4, n_pca_components=5)
        fig = analyzer.plot_results(cube, result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
