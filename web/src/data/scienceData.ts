import type { Module } from "../types";

export const scienceDataModule: Module = {
  id: "science-data",
  title: "Science Data Processing & Visualization",
  subtitle: "Automated pipelines for planetary science data",
  icon: "satellite",
  color: "#3b82f6",
  colorLight: "#93c5fd",
  description:
    "NASA-JPL teams manually download, parse, and visualize science data from missions like Mars rovers, asteroid surveys, and exoplanet telescopes. These workflows involve repetitive API calls, custom parsing scripts, and manual matplotlib figure generation. Devin automates the entire pipeline from data ingestion to publication-ready visualizations.",
  useCases: [
    {
      id: "mars-terrain",
      title: "Mars Rover Terrain Classification",
      situation:
        "Mars Science Laboratory (Curiosity) sends back thousands of navigation camera images per sol. Scientists manually inspect images and classify terrain types for traverse planning. This is time-consuming and subjective.",
      nasaRepo: "nasa-jpl/spoc_lite",
      nasaRepoUrl: "https://github.com/nasa-jpl/spoc_lite",
      before: {
        description:
          "Engineers manually download images from the PDS, visually inspect each frame, and hand-label terrain types in spreadsheets. Gradient analysis and roughness metrics are computed ad-hoc with scattered scripts.",
        code: `# Manual terrain analysis workflow (typical JPL approach)
import requests, numpy as np
from PIL import Image

# Step 1: Manually download each image from PDS
url = "https://mars.nasa.gov/msl-raw-images/..."
response = requests.get(url)
img = Image.open(io.BytesIO(response.content))

# Step 2: Manual gradient computation (copy-pasted per image)
gray = np.array(img.convert('L'), dtype=float)
gx = np.diff(gray, axis=1)  # Simple diff, not robust
gy = np.diff(gray, axis=0)

# Step 3: Eyeball the gradient magnitude
magnitude = np.sqrt(gx[:, :-1]**2 + gy[:-1, :]**2)
print(f"Mean gradient: {magnitude.mean()}")
# ... engineer manually decides terrain type ...

# Step 4: Record in spreadsheet
# "Sol 1000, NAVCAM, rough terrain, gradient=45.2"`,
        painPoints: [
          "Manual image-by-image inspection takes hours per sol",
          "No standardized classification criteria across team members",
          "Gradient analysis scripts are ad-hoc and not version-controlled",
          "Results stored in spreadsheets, not programmatically accessible",
        ],
      },
      devinAction:
        "Devin built a complete MarsSurfaceAnalyzer class that fetches rover images via NASA's Mars Photos API, applies Sobel gradient-based texture analysis with configurable thresholds, automatically classifies terrain into categories (smooth, moderate, rough, hazardous), computes roughness indices, and generates publication-ready terrain maps in a single pipeline call.",
      after: {
        description:
          "Fully automated pipeline: fetch images, classify terrain, generate visualizations. Consistent, reproducible results across the entire team.",
        code: `from science_data_processing import MarsSurfaceAnalyzer

analyzer = MarsSurfaceAnalyzer(api_key="DEMO_KEY")

# Automated: fetch, classify, and visualize in one call
images = analyzer.fetch_rover_images(
    rover="curiosity", sol=1000, camera="NAVCAM"
)

for img in images:
    result = analyzer.classify_terrain(img)
    print(f"Sol {img.sol}: {result.dominant_terrain}")
    print(f"  Roughness: {result.roughness_index:.3f}")
    print(f"  Safe for traverse: {result.is_safe_for_traverse}")

# Generate complete terrain map with all classifications
fig = analyzer.generate_terrain_map(images)
fig.savefig("sol_1000_terrain_map.png", dpi=150)`,
        improvements: [
          "Automated batch processing: entire sol in seconds",
          "Standardized Sobel-gradient classification with configurable thresholds",
          "Reproducible results with version-controlled analysis parameters",
          "Publication-ready matplotlib visualizations generated automatically",
        ],
      },
      value:
        "Reduces terrain classification from hours of manual inspection to seconds of automated analysis. Ensures consistent criteria across the entire science team and produces traceable, reproducible results.",
      metrics: [
        { label: "Time Saved", value: "~95%" },
        { label: "Images/Hour", value: "1000+" },
        { label: "Consistency", value: "100%" },
      ],
    },
    {
      id: "neo-tracking",
      title: "Near-Earth Object Hazard Assessment",
      situation:
        "The Center for Near Earth Object Studies (CNEOS) tracks potentially hazardous asteroids. Engineers query the Sentry API and manually compute risk metrics like the Torino and Palermo scales for each approach.",
      nasaRepo: "NASA CNEOS / Sentry System",
      nasaRepoUrl: "https://cneos.jpl.nasa.gov",
      before: {
        description:
          "Analysts manually query the CNEOS API, download approach data, and compute hazard metrics in separate scripts. Risk categorization is done by hand, and visualization requires custom one-off plotting code.",
        code: `# Manual NEO tracking workflow
import requests, json

# Query CNEOS API
url = "https://api.nasa.gov/neo/rest/v1/feed"
params = {"start_date": "2024-01-01", "api_key": "DEMO_KEY"}
response = requests.get(url, params=params)
data = response.json()

# Manually iterate and extract fields
for date in data["near_earth_objects"]:
    for neo in data["near_earth_objects"][date]:
        name = neo["name"]
        miss_km = float(
            neo["close_approach_data"][0]["miss_distance"]["kilometers"]
        )
        velocity = float(
            neo["close_approach_data"][0]["relative_velocity"]["kph"]
        )
        if neo["is_potentially_hazardous_asteroid"]:
            print(f"HAZARD: {name}, miss={miss_km:.0f} km")
        # No Torino/Palermo scale computation
        # No automated visualization`,
        painPoints: [
          "No automated Torino/Palermo scale computation",
          "Manual iteration over raw JSON responses",
          "No structured hazard categorization (low/medium/high/critical)",
          "Visualization requires separate custom scripts each time",
        ],
      },
      devinAction:
        "Devin built NearEarthObjectTracker with full Torino and Palermo scale estimation, automated hazard categorization, kinetic energy computation, and interactive approach timeline visualizations from a single API call.",
      after: {
        description:
          "Complete NEO tracking pipeline with automated risk assessment and visualization.",
        code: `from science_data_processing import NearEarthObjectTracker

tracker = NearEarthObjectTracker(api_key="DEMO_KEY")

# Fetch and assess all upcoming approaches
neos = tracker.fetch_upcoming_approaches(days=7)

for neo in neos:
    assessment = tracker.assess_hazard(neo)
    print(f"{neo.name}:")
    print(f"  Risk: {assessment.risk_category}")
    print(f"  Torino Scale: {assessment.torino_scale_estimate}")
    print(f"  Palermo Scale: {assessment.palermo_scale_estimate:.2f}")
    print(f"  Kinetic Energy: {assessment.kinetic_energy_mt:.2e} MT")

# Generate interactive approach timeline
fig = tracker.plot_approach_timeline(neos)
fig.savefig("neo_approaches.png", dpi=150)`,
        improvements: [
          "Automated Torino and Palermo scale computation",
          "Structured hazard categorization (low/medium/high/critical)",
          "Kinetic energy estimation from diameter and velocity",
          "Publication-ready approach timeline visualizations",
        ],
      },
      value:
        "Transforms raw API data into actionable risk assessments in milliseconds. Standardizes hazard evaluation across the team and enables real-time monitoring dashboards.",
      metrics: [
        { label: "Risk Metrics", value: "3 scales" },
        { label: "Processing", value: "Real-time" },
        { label: "Coverage", value: "All NEOs" },
      ],
    },
    {
      id: "exoplanet-habitability",
      title: "Exoplanet Habitability Screening",
      situation:
        "The NASA Exoplanet Archive contains 5,000+ confirmed exoplanets. Scientists manually query the TAP service, filter candidates, and compute habitability metrics like the Earth Similarity Index (ESI) using ad-hoc scripts.",
      nasaRepo: "nasa/Kepler-PyKE",
      nasaRepoUrl: "https://github.com/nasa/Kepler-PyKE",
      before: {
        description:
          "Researchers write custom TAP/ADQL queries, manually compute habitable zone boundaries, and estimate ESI with inconsistent formulas across different papers.",
        code: `# Manual exoplanet analysis
import requests, csv, io

# Raw TAP query - error-prone, no validation
tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
query = "SELECT pl_name, pl_rade, pl_eqt, st_lum FROM ps"
response = requests.get(tap_url, params={"query": query, "format": "csv"})

reader = csv.DictReader(io.StringIO(response.text))
for row in reader:
    name = row["pl_name"]
    radius = float(row["pl_rade"]) if row["pl_rade"] else None
    temp = float(row["pl_eqt"]) if row["pl_eqt"] else None

    # Ad-hoc habitability check - inconsistent across team
    if radius and temp:
        if 0.5 < radius < 2.0 and 200 < temp < 350:
            print(f"{name}: possibly habitable?")
    # No ESI computation, no HZ boundary calculation`,
        painPoints: [
          "Raw TAP queries with no error handling or validation",
          "Inconsistent habitability criteria across team members",
          "No Earth Similarity Index (ESI) computation",
          "Manual CSV parsing with no type safety",
        ],
      },
      devinAction:
        "Devin built ExoplanetAnalyzer with proper TAP/ADQL integration, Kopparapu habitable zone boundary computation, multi-parameter Earth Similarity Index, and automated habitability report generation with interactive visualizations.",
      after: {
        description:
          "Systematic habitability screening with standardized ESI computation and publication-ready reports.",
        code: `from science_data_processing import ExoplanetAnalyzer

analyzer = ExoplanetAnalyzer()

# Fetch with proper TAP/ADQL integration
planets = analyzer.fetch_confirmed_exoplanets(limit=500)

# Systematic habitability screening
candidates = analyzer.find_habitable_candidates(planets)
print(f"Found {len(candidates)} HZ candidates from {len(planets)}")

for planet in candidates[:5]:
    esi = analyzer.compute_earth_similarity_index(planet)
    print(f"{planet.name}: ESI={esi:.3f}, T={planet.eq_temp_k:.0f}K")

# Generate full habitability report
report = analyzer.generate_habitability_report(planets)
fig = analyzer.plot_habitability_diagram(planets)
fig.savefig("habitability_report.png", dpi=150)`,
        improvements: [
          "Proper TAP/ADQL integration with error handling",
          "Kopparapu habitable zone boundary computation",
          "Multi-parameter Earth Similarity Index (ESI)",
          "Automated habitability reports with interactive plots",
        ],
      },
      value:
        "Standardizes exoplanet habitability assessment across the science team. Processes the entire catalog in seconds and ensures consistent, reproducible ESI computation.",
      metrics: [
        { label: "Planets Screened", value: "5000+" },
        { label: "ESI Precision", value: "4 params" },
        { label: "Report Gen", value: "< 5 sec" },
      ],
    },
    {
      id: "hyperspectral-mineral",
      title: "Hyperspectral Mineral Mapping",
      situation:
        "JPL's AVIRIS and other imaging spectrometers produce hyperspectral data cubes with 224+ spectral bands. Scientists manually process these cubes to identify mineral compositions using spectral unmixing, PCA, and comparison against spectral libraries.",
      nasaRepo: "nasa-jpl/FlightView",
      nasaRepoUrl: "https://github.com/nasa-jpl/FlightView",
      before: {
        description:
          "Analysts use ENVI or custom MATLAB scripts to process hyperspectral cubes. Each step (calibration, PCA, classification, mineral ID) is a separate manual process with different tools.",
        code: `# Manual hyperspectral processing (typical ENVI/MATLAB workflow)
import numpy as np
from spectral import open_image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Step 1: Load cube (separate tool)
cube = open_image("scene.hdr").load()

# Step 2: Manual PCA (separate script)
reshaped = cube.reshape(-1, cube.shape[2])
pca = PCA(n_components=10)
pca_result = pca.fit_transform(reshaped)
# Manually inspect components...

# Step 3: Classification (another script)
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(pca_result)
# Manually map clusters to minerals...

# Step 4: Mineral ID (yet another script)
# Compare mean spectra against USGS library...
# No automated spectral angle mapping
# No anomaly detection
# Each step is disconnected from the others`,
        painPoints: [
          "Multi-tool workflow: ENVI for loading, Python for PCA, manual mineral ID",
          "No automated spectral angle mapping for mineral identification",
          "No anomaly detection for unusual spectral signatures",
          "Each processing step is disconnected with no integrated pipeline",
        ],
      },
      devinAction:
        "Devin built HyperspectralAnalyzer: a single-class pipeline that handles the entire workflow from spectral cube ingestion through PCA, K-Means classification, spectral angle mineral identification, Mahalanobis anomaly detection, and 6-panel publication-ready visualization.",
      after: {
        description:
          "Fully integrated hyperspectral analysis pipeline from raw cube to mineral map in a single call.",
        code: `from nasa_enhanced_usecases import HyperspectralAnalyzer

analyzer = HyperspectralAnalyzer(n_components=10, n_clusters=8)

# Generate synthetic 224-band hyperspectral scene
cube = analyzer.generate_synthetic_scene(
    n_rows=128, n_cols=128, n_bands=224
)

# Complete analysis pipeline in one call
result = analyzer.analyze(cube)
print(f"Classes: {result.n_classes}")
print(f"Minerals: {result.class_names}")
print(f"Anomalies: {result.anomaly_pixels} pixels flagged")
print(f"PCA variance: {result.explained_variance_ratio:.1%}")

# 6-panel publication figure
fig = analyzer.plot_results(result)
fig.savefig("mineral_map.png", dpi=150)`,
        improvements: [
          "Single-class pipeline replaces multi-tool workflow",
          "Automated spectral angle mapping against mineral library",
          "Mahalanobis distance anomaly detection built-in",
          "6-panel publication-ready visualization in one call",
        ],
      },
      value:
        "Replaces a fragmented multi-tool workflow with a unified Python pipeline. Scientists get from raw data to mineral maps in seconds instead of hours, with reproducible parameters.",
      metrics: [
        { label: "Pipeline Steps", value: "1 call" },
        { label: "Mineral Library", value: "8 endmembers" },
        { label: "Bands Processed", value: "224" },
      ],
    },
  ],
};
