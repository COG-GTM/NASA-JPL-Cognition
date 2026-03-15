import type { Module } from "../types";

export const monteCarloModule: Module = {
  id: "monte-carlo",
  title: "Monte Carlo Simulation & Analysis",
  subtitle: "Uncertainty quantification for mission-critical decisions",
  icon: "chart",
  color: "#10b981",
  colorLight: "#6ee7b7",
  description:
    "Monte Carlo methods are the backbone of mission assurance at JPL. From Entry-Descent-Landing (EDL) dispersion analysis to component reliability and radiation dose estimation, these simulations inform billion-dollar go/no-go decisions. Devin built production-quality Monte Carlo tools that JPL engineers can run immediately.",
  useCases: [
    {
      id: "edl-trajectory",
      title: "EDL Trajectory Uncertainty (Mars Landing)",
      situation:
        "Mars EDL teams run 10,000+ Monte Carlo trajectories to compute landing ellipses and success rates. JPL uses MONTE and custom Fortran codes — both require specialized access and expertise.",
      nasaRepo: "nasa/SMCPy",
      nasaRepoUrl: "https://github.com/nasa/SMCPy",
      before: {
        description:
          "EDL Monte Carlo simulations use MONTE or legacy Fortran codes. Setting up a new analysis requires extensive configuration, and visualizing results requires separate post-processing tools.",
        code: `# EDL Monte Carlo (typical JPL workflow with MONTE)
# Step 1: Write MONTE input deck (proprietary format)
# entry_state.inp, atmosphere.inp, aerodynamics.inp...

# Step 2: Configure Monte Carlo parameters
# Manually specify distributions for each variable:
# - Entry flight path angle: nominal +/- 0.1 deg
# - Entry velocity: nominal +/- 5 m/s
# - Atmospheric density: 1.0 +/- 0.15 (multiplier)
# - Cd variation: +/- 5%
# - Wind: 0 +/- 10 m/s

# Step 3: Submit batch job (hours on cluster)
# montecarlo_run.sh -n 10000 -input entry.inp

# Step 4: Post-process results (separate tool)
# extract_landing_coords.py
# compute_ellipse.py
# plot_dispersion.py

# Each step is a separate tool with different I/O formats
# No integrated pipeline from parameters to visualization`,
        painPoints: [
          "MONTE requires JPL-internal access and specialized training",
          "Separate tools for setup, execution, and post-processing",
          "Batch job submission — hours of wall-clock time on cluster",
          "No integrated parameter-to-visualization pipeline",
        ],
      },
      devinAction:
        "Devin built TrajectoryUncertaintyAnalyzer: a complete EDL Monte Carlo engine with configurable parameter distributions, 6-DOF dispersion computation, 99% landing ellipse estimation, success rate calculation, and integrated visualization — all runnable on a laptop in seconds.",
      after: {
        description:
          "Complete EDL Monte Carlo from parameter specification to landing ellipse in a single Python script.",
        code: `from monte_carlo_simulation import TrajectoryUncertaintyAnalyzer

analyzer = TrajectoryUncertaintyAnalyzer(n_samples=10000, seed=42)

# Run EDL Monte Carlo with configurable parameters
result = analyzer.run_edl_monte_carlo()

print(f"Landing: ({result.nominal_landing_lat:.4f}, "
      f"{result.nominal_landing_lon:.4f})")
print(f"99% ellipse: {result.percentile_99_ellipse_km[0]:.1f} x "
      f"{result.percentile_99_ellipse_km[1]:.1f} km")
print(f"Success rate: {result.success_rate:.1%}")
print(f"Mean downrange: {result.mean_downrange_km:.2f} km")

fig = analyzer.plot_results(result)
fig.savefig("edl_dispersion.png", dpi=150)`,
        improvements: [
          "Runs on a laptop — no cluster or MONTE access needed",
          "Integrated pipeline: parameters -> simulation -> visualization",
          "Configurable distributions for all entry state variables",
          "99% landing ellipse and success rate computed automatically",
        ],
      },
      value:
        "Democratizes EDL dispersion analysis. Engineers can run preliminary landing site assessments on their laptops in seconds, iterating on parameters interactively rather than waiting for cluster batch jobs.",
      metrics: [
        { label: "Samples", value: "10,000+" },
        { label: "Runtime", value: "< 5 sec" },
        { label: "Outputs", value: "Ellipse + stats" },
      ],
    },
    {
      id: "mission-reliability",
      title: "Mission Reliability & Component Failure",
      situation:
        "Mission assurance teams model component-level failure rates to predict overall mission success probability. This analysis uses Weibull distributions, redundancy modeling, and fault tree analysis — typically done in specialized reliability tools like PTC Windchill or ReliaSoft.",
      nasaRepo: "nasa/progpy",
      nasaRepoUrl: "https://github.com/nasa/progpy",
      before: {
        description:
          "Reliability engineers use commercial tools (ReliaSoft, PTC Windchill) or custom MATLAB scripts for mission reliability analysis. Each subsystem failure model is configured separately.",
        code: `# Mission reliability analysis (typical approach)
import numpy as np

n_samples = 10000
rng = np.random.default_rng(42)

# Subsystem 1: Solar arrays (wearout)
shape_solar = 3.5
scale_solar = 100000  # hours
solar_failures = scale_solar * rng.weibull(shape_solar, n_samples)

# Subsystem 2: Reaction wheels (random)
mtbf_rw = 80000
rw_failures = rng.exponential(mtbf_rw, n_samples)

# Subsystem 3: Telecom (copy-paste continues...)
# ... 10+ subsystems manually defined ...

# Manual series/parallel reliability combination
mission_duration = 87660  # 10 years
# Each subsystem checked independently
# No redundancy modeling
# No automated visualization
# No survival curves or availability computation`,
        painPoints: [
          "Commercial tools required (ReliaSoft: $5K+/seat)",
          "Each subsystem manually configured with copy-paste patterns",
          "No built-in redundancy modeling for critical subsystems",
          "Survival curves and availability require separate computation",
        ],
      },
      devinAction:
        "Devin built MissionReliabilitySimulator with pre-configured deep-space subsystem models (solar, telecom, ADCS, propulsion, C&DH, thermal, power), Weibull and exponential failure distributions, configurable redundancy levels, and automated reliability reporting with survival curves.",
      after: {
        description:
          "Complete mission reliability analysis with pre-configured subsystem models and automated reporting.",
        code: `from monte_carlo_simulation import MissionReliabilitySimulator

sim = MissionReliabilitySimulator(n_samples=10000, seed=42)

# Run full reliability analysis (pre-configured subsystems)
result = sim.run_reliability_analysis()

print(f"Mission success: {result.mission_success_rate:.1%}")
print(f"Mean TTF: {result.mean_time_to_first_failure_hours/8766:.1f} yr")
print(f"Availability: {result.availability:.4f}")

# Subsystem-level breakdown
for name, rate in result.subsystem_survival_rates.items():
    status = "PASS" if rate > 0.95 else "REVIEW"
    print(f"  {name}: {rate:.3f} [{status}]")

fig = sim.plot_results(result)
fig.savefig("reliability_analysis.png", dpi=150)`,
        improvements: [
          "Pre-configured deep-space subsystem failure models",
          "Weibull + exponential + radiation SEE failure modes",
          "Built-in redundancy modeling for critical subsystems",
          "Survival curves and availability computed automatically",
        ],
      },
      value:
        "Replaces expensive commercial reliability tools with a free, open-source alternative. Pre-configured subsystem models mean engineers can run preliminary reliability assessments in minutes rather than days of tool setup.",
      metrics: [
        { label: "Subsystems", value: "7 modeled" },
        { label: "Failure Modes", value: "4 types" },
        { label: "Tool Cost", value: "$0" },
      ],
    },
    {
      id: "radiation-environment",
      title: "Radiation Environment & Dose Estimation",
      situation:
        "Radiation engineers estimate Total Ionizing Dose (TID) for missions in Earth orbit, interplanetary space, and jovian environments. This is critical for component selection and shielding design. JPL uses SPENVIS and custom internal tools.",
      nasaRepo: "JPL Design Principles (JPL D-17868)",
      nasaRepoUrl: "https://www.jpl.nasa.gov",
      before: {
        description:
          "Radiation analysis uses SPENVIS (ESA tool), CREME96, or JPL's internal radiation models. Each environment (GCR, SPE, trapped) is modeled separately with different tools.",
        code: `# Radiation environment analysis (typical workflow)
# Step 1: Run SPENVIS for trapped radiation (web tool)
# - Upload orbit parameters
# - Select AP-8/AE-8 models
# - Download dose-depth curves as CSV

# Step 2: Run CREME96 for GCR (separate tool)
# - Configure solar minimum/maximum
# - Select shielding geometry
# - Export dose rates

# Step 3: Run JPL SPE model (internal tool)
# - Select confidence level (90%, 95%)
# - Configure fluence model
# - Compute event probability

# Step 4: Combine manually in Excel
# total_dose = trapped + gcr + spe
# Apply design margin (2x for JPL)
# Compare against component TID ratings
# Manual pass/fail assessment

# Three different tools, three different formats
# No Monte Carlo uncertainty on total dose`,
        painPoints: [
          "Three separate tools for three radiation sources",
          "Manual combination of results in Excel",
          "No Monte Carlo uncertainty on total dose estimate",
          "Design margin assessment done manually",
        ],
      },
      devinAction:
        "Devin built RadiationEnvironmentModel: a unified Monte Carlo radiation environment simulator covering GCR, SPE, and trapped radiation for LEO, interplanetary, and Jupiter missions — with automated shielding analysis, TID margin computation, and design compliance assessment.",
      after: {
        description:
          "Unified radiation environment analysis with Monte Carlo uncertainty quantification.",
        code: `from monte_carlo_simulation import RadiationEnvironmentModel
from monte_carlo_simulation.radiation_environment import (
    RadiationEnvironmentConfig
)

rad = RadiationEnvironmentModel(n_samples=10000, seed=42)

# Jupiter mission (Europa Clipper-like)
config = RadiationEnvironmentConfig(
    environment_type="jupiter",
    mission_duration_years=6.0,
    shielding_mm_al=10.0,
    component_tid_rating_krad=300.0
)

result = rad.run_analysis(config)
print(f"Mean TID: {result.mean_total_dose_krad:.1f} krad")
print(f"95th percentile: {result.percentile_95_dose_krad:.1f} krad")
print(f"Design compliant: {result.design_compliant}")

fig = rad.plot_results(result)
fig.savefig("radiation_environment.png", dpi=150)`,
        improvements: [
          "Unified tool for GCR + SPE + trapped radiation",
          "Monte Carlo uncertainty on total dose estimate",
          "Automated TID margin and design compliance check",
          "Supports LEO, interplanetary, and Jupiter environments",
        ],
      },
      value:
        "Unifies three separate radiation analysis tools into one Monte Carlo framework. Engineers get probabilistic dose estimates with confidence intervals, enabling risk-informed shielding design decisions.",
      metrics: [
        { label: "Sources", value: "GCR+SPE+Trap" },
        { label: "Environments", value: "3 types" },
        { label: "Uncertainty", value: "Full MC" },
      ],
    },
    {
      id: "ice-sheet-mass",
      title: "Ice Sheet Mass Change from Satellite Altimetry",
      situation:
        "JPL's cryosphere team processes satellite altimetry data (ICESat-2, CryoSat-2) to compute ice sheet mass changes for Greenland and Antarctica. The captoolkit provides low-level processing tools, but building a complete mass change pipeline requires significant assembly.",
      nasaRepo: "nasa-jpl/captoolkit",
      nasaRepoUrl: "https://github.com/nasa-jpl/captoolkit",
      before: {
        description:
          "Scientists use captoolkit's individual CLI tools for each processing step: reading HDF5 files, applying corrections, computing crossovers, and fitting trends.",
        code: `# Using nasa-jpl/captoolkit for ice sheet analysis
# Step 1: Read altimetry data
# $ captoolkit_read.py -f icesat2_data.h5 -v lat lon h_li

# Step 2: Apply corrections
# $ captoolkit_corr.py -f data.h5 -c tide ocean inverse_barometer

# Step 3: Compute crossovers
# $ captoolkit_xover.py -f ascending.h5 descending.h5

# Step 4: Fit trends (separate Python script)
import numpy as np
from scipy import stats

times = [...]  # years
dh = [...]     # elevation changes

slope, intercept, r, p, se = stats.linregress(times, dh)
mass_rate = slope * area * density  # Manual conversion
print(f"Mass rate: {mass_rate:.0f} Gt/yr")

# No sea level equivalent computation
# No elevation change map generation
# Each step uses different command-line tools`,
        painPoints: [
          "Multiple CLI tools with different I/O formats for each step",
          "No integrated pipeline from raw altimetry to mass change",
          "Sea level equivalent requires manual conversion",
          "Elevation change maps require separate GIS tools",
        ],
      },
      devinAction:
        "Devin built IceSheetAnalyzer: a complete pipeline from satellite altimetry data to mass change rates, sea level contribution, and elevation change maps — with pre-configured Greenland and Antarctica regions based on published GRACE/GRACE-FO observations.",
      after: {
        description:
          "Integrated ice sheet mass change pipeline with automated reporting and visualization.",
        code: `from nasa_enhanced_usecases import IceSheetAnalyzer
from nasa_enhanced_usecases.cryosphere_analysis.ice_sheet_analysis import (
    GREENLAND, ANTARCTICA
)

analyzer = IceSheetAnalyzer()

# Complete Greenland analysis
result = analyzer.analyze_mass_change(
    region=GREENLAND,
    start_year=2003.0, end_year=2023.0
)

print(f"Mass loss rate: {result.mass_rate_gt_yr:.0f} Gt/yr")
print(f"Acceleration: {result.acceleration_gt_yr2:.1f} Gt/yr^2")
print(f"Total loss: {result.cumulative_mass_change_gt[-1]:.0f} Gt")
print(f"Sea level: {result.sea_level_contribution_mm[-1]:.1f} mm")

fig = analyzer.plot_results(result)
fig.savefig("greenland_mass_change.png", dpi=150)`,
        improvements: [
          "Single-class pipeline replaces multiple CLI tools",
          "Pre-configured regions (Greenland, Antarctica) with published rates",
          "Automated sea level equivalent computation (1 Gt = 1/362 mm)",
          "Elevation change maps with coastal thinning patterns",
        ],
      },
      value:
        "Transforms a fragmented CLI-tool workflow into a unified Python pipeline. Cryosphere scientists get from satellite data to publication figures in one call, with all intermediate computations handled automatically.",
      metrics: [
        { label: "Regions", value: "2 pre-configured" },
        { label: "Pipeline", value: "End-to-end" },
        { label: "SLE Calc", value: "Automatic" },
      ],
    },
  ],
};
