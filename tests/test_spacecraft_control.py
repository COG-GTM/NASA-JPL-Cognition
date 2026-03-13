"""Tests for the spacecraft control simulation module."""

import numpy as np

from spacecraft_control_simulation.attitude_control import (
    SpacecraftAttitudeController,
)
from spacecraft_control_simulation.orbital_mechanics import (
    OrbitalMechanicsSimulator,
)
from spacecraft_control_simulation.propulsion_model import (
    STANDARD_CONFIGS,
    PropulsionSystemModel,
)


class TestSpacecraftAttitudeController:
    def test_simulation_runs(self):
        controller = SpacecraftAttitudeController()
        result = controller.run_simulation(
            initial_attitude_deg=[10, -5, 15],
            target_attitude_deg=[0, 0, 0],
            duration_s=60,
            dt=0.5,
        )
        assert len(result.time) > 0
        assert result.quaternions.shape[1] == 4
        assert result.angular_velocities.shape[1] == 3
        assert result.settling_time is not None
        assert result.max_overshoot_deg is not None

    def test_quaternion_normalization(self):
        controller = SpacecraftAttitudeController()
        result = controller.run_simulation(duration_s=30, dt=0.5)
        norms = np.linalg.norm(result.quaternions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_pointing_error_decreases(self):
        controller = SpacecraftAttitudeController()
        result = controller.run_simulation(
            initial_attitude_deg=[20, -15, 10],
            target_attitude_deg=[0, 0, 0],
            duration_s=300,
            dt=0.5,
        )
        initial_error = result.pointing_errors[0]
        final_error = result.pointing_errors[-1]
        assert final_error < initial_error

    def test_plot_results(self):
        controller = SpacecraftAttitudeController()
        result = controller.run_simulation(duration_s=30, dt=1.0)
        fig = controller.plot_results(result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestOrbitalMechanicsSimulator:
    def test_hohmann_transfer_leo_to_geo(self):
        sim = OrbitalMechanicsSimulator()
        transfer = sim.compute_hohmann_transfer(400, 35786)
        assert transfer.total_delta_v_ms > 3000
        assert transfer.total_delta_v_ms < 5000
        assert transfer.transfer_time_s > 0
        assert transfer.delta_v1_ms > 0
        assert transfer.delta_v2_ms > 0

    def test_orbit_propagation(self):
        sim = OrbitalMechanicsSimulator()
        result = sim.propagate_orbit(
            altitude_km=400,
            inclination_deg=51.6,
            periods=1,
            dt=30.0,
        )
        assert len(result.time) > 0
        assert result.position.shape[1] == 3
        assert result.velocity.shape[1] == 3
        # Altitude should stay near 400 km for circular orbit
        mean_alt = np.mean(result.altitude_km)
        assert abs(mean_alt - 400) < 50

    def test_ground_track_bounds(self):
        sim = OrbitalMechanicsSimulator()
        result = sim.propagate_orbit(
            altitude_km=400,
            inclination_deg=51.6,
            periods=1,
            dt=30.0,
        )
        assert np.all(result.latitude_deg >= -90)
        assert np.all(result.latitude_deg <= 90)
        assert np.all(result.longitude_deg >= -180)
        assert np.all(result.longitude_deg <= 180)

    def test_plot_ground_track(self):
        sim = OrbitalMechanicsSimulator()
        result = sim.propagate_orbit(altitude_km=400, periods=1, dt=60.0)
        fig = sim.plot_ground_track(result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPropulsionSystemModel:
    def test_propellant_budget(self):
        model = PropulsionSystemModel()
        config = STANDARD_CONFIGS["biprop_main"]
        budget = model.compute_propellant_budget(
            delta_v_ms=2000.0,
            spacecraft_dry_mass_kg=500.0,
            config=config,
        )
        assert budget.propellant_mass_kg > 0
        assert budget.total_mass_kg > budget.spacecraft_dry_mass_kg
        assert budget.mass_ratio > 1.0
        assert budget.burn_time_s > 0

    def test_tsiolkovsky_equation(self):
        model = PropulsionSystemModel()
        config = STANDARD_CONFIGS["biprop_main"]
        budget = model.compute_propellant_budget(
            delta_v_ms=0.0,
            spacecraft_dry_mass_kg=500.0,
            config=config,
        )
        assert abs(budget.propellant_mass_kg) < 0.01
        assert abs(budget.mass_ratio - 1.0) < 0.01

    def test_trade_study(self):
        model = PropulsionSystemModel()
        result = model.run_trade_study(delta_v_ms=2000.0, dry_mass_kg=500.0)
        assert len(result.configs) == len(STANDARD_CONFIGS)
        assert len(result.budgets) == len(STANDARD_CONFIGS)
        assert result.optimal_config is not None
        assert result.optimal_budget is not None

    def test_ion_engine_lower_propellant(self):
        model = PropulsionSystemModel()
        biprop_budget = model.compute_propellant_budget(
            2000.0, 500.0, STANDARD_CONFIGS["biprop_main"]
        )
        ion_budget = model.compute_propellant_budget(2000.0, 500.0, STANDARD_CONFIGS["ion_nstar"])
        assert ion_budget.propellant_mass_kg < biprop_budget.propellant_mass_kg

    def test_plot_trade_study(self):
        model = PropulsionSystemModel()
        result = model.run_trade_study(delta_v_ms=1000.0, dry_mass_kg=300.0)
        fig = model.plot_trade_study(result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
