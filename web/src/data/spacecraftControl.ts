import type { Module } from "../types";

export const spacecraftControlModule: Module = {
  id: "spacecraft-control",
  title: "Spacecraft Control Simulation",
  subtitle: "Python-based dynamics & control — open-source Simulink alternative",
  icon: "rocket",
  color: "#8b5cf6",
  colorLight: "#c4b5fd",
  description:
    "JPL engineers traditionally use MATLAB/Simulink for spacecraft dynamics and control simulation. These licenses are expensive, the models aren't easily version-controlled, and collaboration is limited. Devin built Python-based equivalents that are open-source, Git-friendly, and produce identical physics fidelity.",
  useCases: [
    {
      id: "attitude-control",
      title: "Spacecraft Attitude Control Simulation",
      situation:
        "Attitude Determination and Control Systems (ADCS) engineers design and validate pointing controllers using Simulink. The models are binary files that can't be diffed, code-reviewed, or easily shared.",
      nasaRepo: "nasa-jpl/SAAS",
      nasaRepoUrl: "https://github.com/nasa-jpl/SAAS",
      before: {
        description:
          "Engineers build Simulink block diagrams for attitude control, then manually run simulations and inspect plots. Model files are binary (.slx), making version control and code review impossible.",
        code: `% MATLAB/Simulink attitude control (typical JPL workflow)
% Step 1: Open Simulink model (binary .slx file)
open_system('spacecraft_adcs.slx');

% Step 2: Set parameters manually in GUI
set_param('spacecraft_adcs/PID', 'P', '2.0');
set_param('spacecraft_adcs/PID', 'I', '0.01');
set_param('spacecraft_adcs/PID', 'D', '5.0');

% Step 3: Set initial conditions in workspace
J = diag([500, 600, 400]);  % Inertia tensor
q0 = [0.1, -0.05, 0.15, 0.98];  % Initial quaternion

% Step 4: Run simulation
sim('spacecraft_adcs', 600);

% Step 5: Manually extract and plot results
figure; plot(tout, yout(:, 1:3));
title('Euler Angles');
% Manual inspection of settling time, overshoot...
% Problems: binary files, expensive license, no CI/CD`,
        painPoints: [
          "Binary .slx files — impossible to diff, review, or merge",
          "MATLAB/Simulink licenses cost $10K+ per seat per year",
          "Cannot run in CI/CD pipelines or automated testing",
          "Parameter sweeps require manual GUI interaction",
        ],
      },
      devinAction:
        "Devin built SpacecraftAttitudeController: quaternion-based 3-DOF attitude dynamics with full Euler's equations, PID + reaction wheel control, configurable spacecraft inertia, and automated settling time / pointing error analysis — all in pure Python.",
      after: {
        description:
          "Pure Python attitude control simulation with quaternion kinematics, fully testable and version-controllable.",
        code: `from spacecraft_control_simulation import SpacecraftAttitudeController

controller = SpacecraftAttitudeController(
    inertia_kg_m2=[500.0, 600.0, 400.0],
    kp=2.0, ki=0.01, kd=5.0,
    max_torque_nm=1.0
)

# Run simulation with full quaternion dynamics
result = controller.run_simulation(
    initial_attitude_deg=[15, -10, 20],
    target_attitude_deg=[0, 0, 0],
    duration_s=600, dt_s=0.1
)

print(f"Settling time: {result.settling_time:.1f} s")
print(f"Max overshoot: {result.max_overshoot_deg:.2f} deg")
print(f"Steady-state error: {result.steady_state_error_deg:.4f} deg")

fig = controller.plot_results(result)
fig.savefig("attitude_sim.png", dpi=150)`,
        improvements: [
          "Pure Python — no MATLAB license needed ($10K+/yr savings)",
          "Text-based code — full git diff, review, and merge support",
          "Runs in CI/CD — automated regression testing of control laws",
          "Programmatic parameter sweeps for trade studies",
        ],
      },
      value:
        "Eliminates MATLAB/Simulink dependency for ADCS simulation. Engineers can version-control their models, run automated tests in CI, and collaborate via pull requests — just like flight software.",
      metrics: [
        { label: "License Cost", value: "$0" },
        { label: "Physics Fidelity", value: "3-DOF" },
        { label: "CI Compatible", value: "Yes" },
      ],
    },
    {
      id: "orbital-mechanics",
      title: "Orbital Mechanics & Transfer Design",
      situation:
        "Mission designers use JPL's MONTE (Mission Analysis, Operations, and Navigation Toolkit Environment) for trajectory design. MONTE is powerful but has a steep learning curve and limited accessibility outside JPL.",
      nasaRepo: "nasa-jpl/MonteCop",
      nasaRepoUrl: "https://github.com/nasa-jpl/MonteCop",
      before: {
        description:
          "Engineers use MONTE or STK for orbital mechanics, requiring specialized training and expensive licenses. Simple Hohmann transfers and orbit propagation require complex setup.",
        code: `# Typical trajectory design workflow (MONTE/STK)
import Monte
from Monte import TrajDesign

# Complex initialization
universe = Monte.Universe()
earth = universe.addBody("Earth", GM=3.986e14)

# Define spacecraft
sc = TrajDesign.Spacecraft()
sc.setInitialState(
    epoch="2024-06-15T00:00:00",
    sma=6778.0, ecc=0.001, inc=28.5,
    # ... 3 more orbital elements
)

# Compute transfer (complex API)
transfer = TrajDesign.HohmannTransfer(
    departure=sc.orbit,
    arrival=TrajDesign.CircularOrbit(sma=42164.0)
)
# Manual extraction of delta-V, transfer time...
# Ground track requires separate tool (STK)
# J2 perturbation is another separate module`,
        painPoints: [
          "MONTE requires specialized training and JPL-internal access",
          "Expensive commercial alternatives (STK, GMAT) for external teams",
          "Ground track visualization requires separate tool (STK)",
          "J2 perturbation analysis is a separate module with different API",
        ],
      },
      devinAction:
        "Devin built OrbitalMechanicsSimulator: Keplerian propagation with J2 perturbation, Hohmann transfer computation, ground track generation, and orbital element visualization — all accessible through a clean, intuitive API.",
      after: {
        description:
          "Accessible orbital mechanics toolkit with integrated visualization and perturbation modeling.",
        code: `from spacecraft_control_simulation import OrbitalMechanicsSimulator
from spacecraft_control_simulation.orbital_mechanics import OrbitalElements

sim = OrbitalMechanicsSimulator()

# Hohmann transfer: LEO to GEO
transfer = sim.compute_hohmann_transfer(r1_km=400, r2_km=35786)
print(f"Delta-V1: {transfer.delta_v1_ms:.1f} m/s")
print(f"Delta-V2: {transfer.delta_v2_ms:.1f} m/s")
print(f"Total: {transfer.total_delta_v_ms:.1f} m/s")
print(f"Transfer time: {transfer.transfer_time_s/3600:.1f} hours")

# Propagate orbit with J2 perturbation
orbit = OrbitalElements(
    semi_major_axis_km=6778.0, eccentricity=0.001,
    inclination_deg=51.6  # ISS-like
)
states = sim.propagate_orbit(orbit, duration_hours=24.0)

fig = sim.plot_ground_track(states)
fig.savefig("ground_track.png", dpi=150)`,
        improvements: [
          "Zero-cost, open-source alternative to MONTE/STK",
          "Integrated J2 perturbation in the same propagator",
          "Built-in ground track visualization (no separate tool)",
          "Clean API — minimal learning curve for new engineers",
        ],
      },
      value:
        "Democratizes orbital mechanics computation. Any engineer can design transfers and visualize orbits without specialized tools or training. Perfect for early mission design and trade studies.",
      metrics: [
        { label: "Setup Time", value: "< 1 min" },
        { label: "Perturbations", value: "J2" },
        { label: "Transfer Types", value: "Hohmann" },
      ],
    },
    {
      id: "propulsion-trade",
      title: "Propulsion System Trade Studies",
      situation:
        "Propulsion engineers evaluate multiple engine configurations for mission delta-V requirements. This involves Tsiolkovsky calculations, propellant budgeting, and multi-parameter trade studies — typically done in Excel spreadsheets.",
      nasaRepo: "Dawn, Cassini, MRO propulsion specs",
      nasaRepoUrl: "https://www.jpl.nasa.gov/missions",
      before: {
        description:
          "Trade studies are done in Excel with manual Tsiolkovsky calculations. Each propulsion option requires a separate row with hand-computed propellant masses, burn times, and performance metrics.",
        code: `# Typical propulsion trade study (Excel-based workflow)
import math

g0 = 9.80665  # m/s^2

# Config 1: Hydrazine monoprop
isp_1 = 230  # s
delta_v = 2000  # m/s
dry_mass = 500  # kg
mass_ratio_1 = math.exp(delta_v / (isp_1 * g0))
prop_mass_1 = dry_mass * (mass_ratio_1 - 1)
print(f"Hydrazine: {prop_mass_1:.0f} kg propellant")

# Config 2: Biprop (copy-paste, change Isp)
isp_2 = 320
mass_ratio_2 = math.exp(delta_v / (isp_2 * g0))
prop_mass_2 = dry_mass * (mass_ratio_2 - 1)

# Config 3: Ion engine (copy-paste again)
isp_3 = 3100
mass_ratio_3 = math.exp(delta_v / (isp_3 * g0))
prop_mass_3 = dry_mass * (mass_ratio_3 - 1)

# No visualization, no burn time, no trade matrix
# Results pasted into PowerPoint for review`,
        painPoints: [
          "Copy-paste Tsiolkovsky calculations for each configuration",
          "No systematic comparison or trade matrix generation",
          "Results manually transferred to PowerPoint for reviews",
          "No burn time or thrust-to-weight computation",
        ],
      },
      devinAction:
        "Devin built PropulsionSystemModel with pre-configured real spacecraft propulsion systems (Dawn ion, Cassini biprop, MRO monoprop), automated trade study generation, and 4-panel comparison visualizations.",
      after: {
        description:
          "Automated propulsion trade studies with real spacecraft configurations and publication-ready visualizations.",
        code: `from spacecraft_control_simulation import PropulsionSystemModel

prop = PropulsionSystemModel()

# Automated trade study across all configurations
trade = prop.run_trade_study(
    delta_v_ms=2000.0, dry_mass_kg=500.0
)

# Automatic optimal selection
print(f"Optimal: {trade.optimal_config.name}")
print(f"  Isp: {trade.optimal_config.specific_impulse_s} s")
print(f"  Propellant: {trade.optimal_budget.propellant_mass_kg:.1f} kg")
print(f"  Burn time: {trade.optimal_budget.burn_time_s:.0f} s")

# Compare all configurations
for config, budget in zip(trade.configs, trade.budgets):
    print(f"{config.name}: {budget.propellant_mass_kg:.0f} kg")

fig = prop.plot_trade_study(trade)
fig.savefig("propulsion_trade.png", dpi=150)`,
        improvements: [
          "Pre-configured real spacecraft propulsion systems",
          "Automated trade matrix with optimal selection",
          "Burn time and thrust computation included",
          "4-panel publication-ready comparison figures",
        ],
      },
      value:
        "Replaces error-prone Excel spreadsheets with reproducible, automated trade studies. New propulsion configurations can be added in seconds and the entire study re-run instantly.",
      metrics: [
        { label: "Configs", value: "5 built-in" },
        { label: "Metrics", value: "6 per config" },
        { label: "Study Time", value: "< 1 sec" },
      ],
    },
    {
      id: "battery-rul",
      title: "Spacecraft Battery Remaining Useful Life",
      situation:
        "Deep-space missions rely on batteries for critical operations. NASA's progpy framework provides prognostics tools, but building a complete RUL prediction pipeline from raw cycling data requires significant custom code.",
      nasaRepo: "nasa/progpy",
      nasaRepoUrl: "https://github.com/nasa/progpy",
      before: {
        description:
          "Engineers use progpy's BatteryElectroChemEOD model with manual parameter tuning. Building a complete pipeline from cycling data to RUL prediction with uncertainty bounds requires extensive custom code.",
        code: `# Using nasa/progpy for battery RUL
from progpy.models import BatteryElectroChemEOD
from scipy.optimize import minimize

# Step 1: Initialize model with default params
batt = BatteryElectroChemEOD()

# Step 2: Manual parameter estimation
# (requires separate optimization loop)
def objective(params):
    batt.parameters['qMax'] = params[0]
    # ... manual parameter mapping
    # ... run simulation, compare to data
    pass

# Step 3: Run prediction
# progpy provides the framework but not the
# end-to-end pipeline for cycling data analysis
# No multi-model fitting comparison
# No automated uncertainty quantification
# No publication-ready visualization`,
        painPoints: [
          "Manual parameter estimation with custom optimization loops",
          "No built-in multi-model degradation comparison",
          "Uncertainty quantification requires separate implementation",
          "No end-to-end pipeline from cycling data to RUL prediction",
        ],
      },
      devinAction:
        "Devin built BatteryRULPredictor: an end-to-end pipeline that fits 3 degradation models (exponential, power-law, double-exponential), automatically selects the best fit, computes RUL with 5th-95th percentile uncertainty bounds, and generates 4-panel diagnostic visualizations.",
      after: {
        description:
          "Complete battery RUL pipeline from cycling data to prediction with uncertainty bounds.",
        code: `from nasa_enhanced_usecases import BatteryRULPredictor

predictor = BatteryRULPredictor()

# Generate realistic cycling data (or load real data)
data = predictor.generate_synthetic_battery_data(n_cycles=500)

# End-to-end RUL prediction
prediction = predictor.predict_rul(data, eol_threshold_ah=1.4)
print(f"Current capacity: {prediction.current_capacity_ah:.2f} Ah")
print(f"RUL: {prediction.predicted_rul_cycles:.0f} cycles")
print(f"Confidence: {prediction.confidence:.0%}")
print(f"Best model: {prediction.best_model}")
print(f"95th percentile: {prediction.rul_upper_bound:.0f} cycles")

fig = predictor.plot_prediction(prediction)
fig.savefig("battery_rul.png", dpi=150)`,
        improvements: [
          "End-to-end pipeline: data -> fit -> predict -> visualize",
          "3 degradation models compared automatically (best-fit selection)",
          "5th-95th percentile uncertainty bounds on RUL estimate",
          "4-panel diagnostic visualization generated in one call",
        ],
      },
      value:
        "Transforms battery health monitoring from manual analysis into an automated prediction pipeline. Mission planners get real-time RUL estimates with confidence intervals for critical power budget decisions.",
      metrics: [
        { label: "Models Fit", value: "3 auto" },
        { label: "Uncertainty", value: "5-95%" },
        { label: "Pipeline", value: "1 call" },
      ],
    },
  ],
};
