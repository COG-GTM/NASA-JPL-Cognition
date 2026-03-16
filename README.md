# NASA-JPL Science & Engineering Toolkit

A comprehensive Python toolkit for science data processing, spacecraft control simulation, and Monte Carlo analysis — built to demonstrate how AI-assisted development accelerates workflows that NASA-JPL engineers perform daily.

Each module references real NASA/NASA-JPL open-source repositories and implements simplified but realistic versions of the workflows found in those tools.

## Modules

### 1. Science Data Processing & Visualization

Automated pipelines for processing and visualizing planetary science data from NASA's public APIs.

| Component | Description | NASA API / Reference |
|-----------|-------------|---------------------|
| **Mars Surface Analysis** | Terrain classification from rover imagery using gradient-based texture analysis | [NASA Mars Photos API](https://api.nasa.gov), [nasa-jpl/spoc_lite](https://github.com/nasa-jpl/spoc_lite) |
| **Near-Earth Object Tracking** | Real-time asteroid tracking with hazard assessment (Palermo/Torino scales) | [NASA CNEOS API](https://api.nasa.gov/neo), [NASA Center for NEO Studies](https://cneos.jpl.nasa.gov) |
| **Exoplanet Analysis** | Habitability screening with Earth Similarity Index computation | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu), [nasa/Kepler-PyKE](https://github.com/nasa/Kepler-PyKE) |

```python
from science_data_processing import MarsSurfaceAnalyzer, NearEarthObjectTracker, ExoplanetAnalyzer

# Mars terrain classification
analyzer = MarsSurfaceAnalyzer(api_key="DEMO_KEY")
images = analyzer.fetch_rover_images(rover="curiosity", sol=1000, camera="NAVCAM")
for img in images[:5]:
    classification = analyzer.classify_terrain(img)
    print(f"Sol {img.sol}: {classification.dominant_terrain} ({classification.roughness_index:.3f})")

# Asteroid tracking
tracker = NearEarthObjectTracker()
neos = tracker.fetch_upcoming_approaches(days=7)
for neo in neos:
    assessment = tracker.assess_hazard(neo)
    print(f"{neo.name}: {assessment.risk_category} (Torino {assessment.torino_scale_estimate})")

# Exoplanet habitability
exo = ExoplanetAnalyzer()
planets = exo.fetch_confirmed_exoplanets(limit=500)
candidates = exo.find_habitable_candidates(planets)
print(f"Found {len(candidates)} habitable zone candidates")
```

### 2. Spacecraft Control Simulation

Python-based spacecraft dynamics and control — the open-source equivalent of Simulink models — with full 3-DOF attitude dynamics, orbital mechanics, and propulsion system modeling.

| Component | Description | Reference |
|-----------|-------------|-----------|
| **Attitude Control** | Quaternion-based 3-DOF dynamics with PID + reaction wheel control | [nasa-jpl/SAAS](https://github.com/nasa-jpl/SAAS) |
| **Orbital Mechanics** | Keplerian propagation with J2 perturbation, Hohmann transfers | [nasa-jpl/MonteCop](https://github.com/nasa-jpl/MonteCop), JPL MONTE |
| **Propulsion Modeling** | Tsiolkovsky rocket equation, multi-system trade studies | Based on real spacecraft specs (Dawn, Cassini, MRO) |

```python
from spacecraft_control_simulation import (
    SpacecraftAttitudeController, OrbitalMechanicsSimulator, PropulsionSystemModel
)

# Attitude control simulation
controller = SpacecraftAttitudeController()
result = controller.run_simulation(
    initial_attitude_deg=[15, -10, 20],
    target_attitude_deg=[0, 0, 0],
    duration_s=600,
)
print(f"Settling time: {result.settling_time:.1f} s")
print(f"Steady-state error: {result.steady_state_error_deg:.4f} deg")
controller.plot_results(result, output_path="attitude_sim.png")

# Orbital mechanics
sim = OrbitalMechanicsSimulator()
transfer = sim.compute_hohmann_transfer(400, 35786)  # LEO to GEO
print(f"Total delta-V: {transfer.total_delta_v_ms:.1f} m/s")
print(f"Transfer time: {transfer.transfer_time_s/3600:.2f} hours")

# Propulsion trade study
prop = PropulsionSystemModel()
trade = prop.run_trade_study(delta_v_ms=2000.0, dry_mass_kg=500.0)
print(f"Optimal: {trade.optimal_config.name}")
print(f"Propellant: {trade.optimal_budget.propellant_mass_kg:.1f} kg")
```

### 3. Monte Carlo Simulation

Monte Carlo methods for spacecraft mission analysis — trajectory uncertainty, mission reliability, and radiation environment modeling.

| Component | Description | Reference |
|-----------|-------------|-----------|
| **Trajectory Uncertainty** | EDL dispersion analysis with 99% landing ellipse computation | [nasa/SMCPy](https://github.com/nasa/SMCPy), JPL MONTE |
| **Mission Reliability** | Component-level failure simulation with Weibull/exponential models | [nasa/progpy](https://github.com/nasa/progpy), NASA-STD-8729.1 |
| **Radiation Environment** | GCR, SPE, and trapped radiation dose accumulation modeling | JPL Design Principles (JPL D-17868) |

```python
from monte_carlo_simulation import (
    TrajectoryUncertaintyAnalyzer, MissionReliabilitySimulator, RadiationEnvironmentModel
)

# EDL Monte Carlo (Mars landing dispersion)
analyzer = TrajectoryUncertaintyAnalyzer(n_samples=10000)
result = analyzer.run_edl_monte_carlo()
print(f"99% landing ellipse: {result.percentile_99_ellipse_km[0]:.1f} x "
      f"{result.percentile_99_ellipse_km[1]:.1f} km")
print(f"Success rate: {result.success_rate:.1%}")

# Mission reliability (deep-space mission)
sim = MissionReliabilitySimulator(n_samples=10000)
result = sim.run_reliability_analysis()
print(f"Mission success probability: {result.mission_success_rate:.3f}")
print(f"Mean time to failure: {result.mean_time_to_first_failure_hours/8766:.1f} years")

# Radiation environment (Europa Clipper-like)
from monte_carlo_simulation.radiation_environment import RadiationEnvironmentConfig
rad = RadiationEnvironmentModel(n_samples=10000)
config = RadiationEnvironmentConfig(environment_type="jupiter", mission_duration_years=6.0)
result = rad.run_analysis(config)
print(f"Mean TID: {result.mean_total_dose_krad:.1f} krad")
print(f"TID margin: {result.tid_margin:.1f} krad")
```

### 4. NASA Enhanced Use Cases

Real workflows pulled from NASA/NASA-JPL open-source repositories, enhanced with automated processing, prediction, and visualization.

| Component | Original Repo | Enhancement |
|-----------|--------------|-------------|
| **Battery RUL Prediction** | [nasa/progpy](https://github.com/nasa/progpy) (118★) | Automated multi-model degradation fitting, real-time RUL with confidence intervals |
| **Ice Sheet Mass Change** | [nasa-jpl/captoolkit](https://github.com/nasa-jpl/captoolkit) (76★) | Full pipeline from altimetry to corrections to mass change to sea level contribution |
| **Hyperspectral Analysis** | [nasa-jpl/FlightView](https://github.com/nasa-jpl/FlightView) (29★) | Automated spectral unmixing, mineral ID, anomaly detection from imaging spectrometers |

```python
from nasa_enhanced_usecases import BatteryRULPredictor, IceSheetAnalyzer, HyperspectralAnalyzer

# Battery remaining useful life
predictor = BatteryRULPredictor()
data = predictor.generate_synthetic_battery_data(n_cycles=500)
prediction = predictor.predict_rul(data, eol_threshold_ah=1.4)
print(f"RUL: {prediction.predicted_rul_cycles:.0f} cycles ({prediction.confidence:.0%} confidence)")

# Ice sheet mass change (Greenland)
from nasa_enhanced_usecases.cryosphere_analysis.ice_sheet_analysis import GREENLAND
ice = IceSheetAnalyzer()
result = ice.analyze_mass_change(region=GREENLAND, start_year=2003.0, end_year=2023.0)
print(f"Mass loss rate: {result.mass_rate_gt_yr:.0f} Gt/yr")
print(f"Sea level contribution: {result.sea_level_contribution_mm[-1]:.1f} mm")

# Hyperspectral mineral mapping
spectral = HyperspectralAnalyzer()
cube = spectral.generate_synthetic_scene()
result = spectral.analyze(cube)
print(f"Identified {result.n_classes} spectral classes: {result.class_names}")
```

## Interactive Web UI

The toolkit includes a browser-based showcase that walks you through each module with interactive before/after comparisons showing how Devin accelerates NASA-JPL workflows.

### Quick Start (Web UI)

```bash
# Clone the repo
git clone https://github.com/COG-GTM/NASA-JPL-Cognition.git
cd NASA-JPL-Cognition

# Start the web UI
cd web
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

The web UI shows **12 use cases** across 3 modules, each with:
- **Situation**: The real NASA-JPL problem
- **Before**: The manual workflow with pain points
- **What Devin Built**: The automated solution
- **After**: The enhanced code with improvements
- **Value**: Business impact and metrics

### Building for Production

```bash
cd web
npm run build    # Output in web/dist/
npm run preview  # Preview production build
```

## Python Toolkit Installation

```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- numpy, scipy, matplotlib, pandas, astropy, scikit-learn, plotly, h5py, requests, tqdm

## Testing

```bash
PYTHONPATH=. MPLBACKEND=Agg pytest tests/ -v
```

All 48 tests pass across 4 modules.

## Related NASA/NASA-JPL Open-Source Tools

This toolkit draws inspiration from the following NASA repositories:

| Repository | Stars | Description |
|-----------|-------|-------------|
| [nasa-jpl/autoRIFT](https://github.com/nasa-jpl/autoRIFT) | 260 | Dense pixel displacement algorithms for ice velocity |
| [nasa/SMCPy](https://github.com/nasa/SMCPy) | 126 | Sequential Monte Carlo sampling for Bayesian inference |
| [nasa/progpy](https://github.com/nasa/progpy) | 118 | Prognostic Python Packages for remaining useful life |
| [nasa-jpl/captoolkit](https://github.com/nasa-jpl/captoolkit) | 76 | Cryosphere Altimetry Processing Toolkit |
| [nasa/harmony-py](https://github.com/nasa/harmony-py) | 68 | Python library for NASA Harmony data transformation |
| [nasa-jpl/its_live](https://github.com/nasa-jpl/its_live) | 53 | Global glacier velocity datasets |
| [nasa-jpl/FlightView](https://github.com/nasa-jpl/FlightView) | 29 | Real-time imaging spectroscopy tools |
| [nasa-jpl/MonteCop](https://github.com/nasa-jpl/MonteCop) | 11 | Monte/Copernicus trajectory interoperability |
| [nasa-jpl/SAAS](https://github.com/nasa-jpl/SAAS) | 6 | System-level autonomy simulation |

## License

Apache 2.0   
