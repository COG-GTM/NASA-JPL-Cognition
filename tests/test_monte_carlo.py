"""Tests for the Monte Carlo simulation module."""

from monte_carlo_simulation.mission_reliability import (
    DEEP_SPACE_SUBSYSTEMS,
    MissionConfig,
    MissionReliabilitySimulator,
)
from monte_carlo_simulation.radiation_environment import (
    RadiationEnvironmentConfig,
    RadiationEnvironmentModel,
)
from monte_carlo_simulation.trajectory_uncertainty import (
    TrajectoryUncertaintyAnalyzer,
)


class TestTrajectoryUncertaintyAnalyzer:
    def test_edl_monte_carlo(self):
        analyzer = TrajectoryUncertaintyAnalyzer(n_samples=200, seed=42)
        result = analyzer.run_edl_monte_carlo()
        assert result.n_samples == 200
        assert len(result.landing_latitude_deg) == 200
        assert len(result.landing_longitude_deg) == 200
        assert result.mean_landing_error_km >= 0
        assert result.success_rate >= 0
        assert result.success_rate <= 1

    def test_orbit_insertion_monte_carlo(self):
        analyzer = TrajectoryUncertaintyAnalyzer(n_samples=200, seed=42)
        result = analyzer.run_orbit_insertion_monte_carlo(
            nominal_delta_v_ms=1000.0,
            target_altitude_km=400.0,
        )
        assert result.n_samples == 200
        assert len(result.landing_altitude_km) == 200
        assert result.success_rate >= 0

    def test_ellipse_dimensions(self):
        analyzer = TrajectoryUncertaintyAnalyzer(n_samples=500, seed=42)
        result = analyzer.run_edl_monte_carlo()
        assert result.percentile_99_ellipse_km[0] >= 0
        assert result.percentile_99_ellipse_km[1] >= 0

    def test_plot_results(self):
        analyzer = TrajectoryUncertaintyAnalyzer(n_samples=200, seed=42)
        result = analyzer.run_edl_monte_carlo()
        fig = analyzer.plot_results(result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestMissionReliabilitySimulator:
    def test_reliability_analysis(self):
        sim = MissionReliabilitySimulator(n_samples=500, seed=42)
        result = sim.run_reliability_analysis()
        assert result.n_samples == 500
        assert 0 <= result.mission_success_rate <= 1
        assert result.mean_time_to_first_failure_hours > 0
        assert result.availability > 0
        assert result.availability <= 1

    def test_custom_mission(self):
        sim = MissionReliabilitySimulator(n_samples=200, seed=42)
        config = MissionConfig(
            name="Test Mission",
            duration_hours=365.25 * 24,  # 1 year
            subsystems=DEEP_SPACE_SUBSYSTEMS[:3],
        )
        result = sim.run_reliability_analysis(config)
        assert result.mission_config.name == "Test Mission"
        assert len(result.subsystem_survival_rates) == 3

    def test_subsystem_survival_rates(self):
        sim = MissionReliabilitySimulator(n_samples=500, seed=42)
        result = sim.run_reliability_analysis()
        for name, rate in result.subsystem_survival_rates.items():
            assert 0 <= rate <= 1, f"{name} has invalid survival rate: {rate}"

    def test_plot_results(self):
        sim = MissionReliabilitySimulator(n_samples=200, seed=42)
        result = sim.run_reliability_analysis()
        fig = sim.plot_results(result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRadiationEnvironmentModel:
    def test_radiation_analysis(self):
        model = RadiationEnvironmentModel(n_samples=500, seed=42)
        result = model.run_analysis()
        assert result.n_samples == 500
        assert result.mean_total_dose_krad > 0
        assert result.dose_99_percentile_krad >= result.mean_total_dose_krad
        assert result.see_rate_per_day >= 0

    def test_jupiter_environment(self):
        model = RadiationEnvironmentModel(n_samples=200, seed=42)
        config = RadiationEnvironmentConfig(
            environment_type="jupiter",
            mission_duration_years=6.0,
        )
        result = model.run_analysis(config)
        assert result.mean_total_dose_krad > 0

    def test_shielding_reduces_dose(self):
        model = RadiationEnvironmentModel(n_samples=500, seed=42)
        thin_config = RadiationEnvironmentConfig(shielding_thickness_mm_al=1.0)
        thick_config = RadiationEnvironmentConfig(shielding_thickness_mm_al=10.0)
        thin_result = model.run_analysis(thin_config)
        # Re-init to reset RNG
        model2 = RadiationEnvironmentModel(n_samples=500, seed=42)
        thick_result = model2.run_analysis(thick_config)
        assert thick_result.mean_total_dose_krad < thin_result.mean_total_dose_krad

    def test_plot_results(self):
        model = RadiationEnvironmentModel(n_samples=200, seed=42)
        result = model.run_analysis()
        fig = model.plot_results(result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
