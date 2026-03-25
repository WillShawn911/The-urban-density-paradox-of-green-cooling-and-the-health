The Urban Density Paradox of Green Cooling and the Health Dividend of Targeted Acupuncture Across 118 Cities

📌 Overview

This repository contains the complete analytical pipeline, high-resolution spatial datasets, and simulation models for our global study on urban heat-health equity.

We investigate the "Urban Density Paradox"—the critical tipping point where high building density neutralizes the cooling efficacy of green spaces. By analyzing 11.2 million neighborhood grids across 118 global cities, we demonstrate how "urban acupuncture" (targeted pocket parks) can reduce thermal injustice significantly more effectively than traditional large-scale park planning.

📂 Repository Structure

1. /data - Processed Results

These files contain the final analytical outputs used to generate the figures and tables in the manuscript:

Global_118_Cities_Equity_Metrics.csv: City-level equity profiles including total population, baseline Gini coefficients, and the percentage of residents with zero green access.

Global_Micro_Grid_Thermal_Data.csv: Micro-grid level data (500m resolution) integrating accessibility scores, Urban Heat Island Intensity (UHII), and Nighttime Lights (NTL) as a wealth proxy.

All_118_Cities_Simulation_Results.csv: Quantitative comparison of Gini coefficient shifts across three intervention scenarios (Mega-Park, Equal Greening, and Pocket Parks).

2. /scripts - Analytical Pipeline

The scripts are numbered by their logical execution order:

01_Accessibility_2SFCA.py: Implementation of the modified 2-Step Floating Catchment Area (2SFCA) method, incorporating NDVI-weighted park quality and street-network walking distances.

02_Equity_and_ml_attribution.py: Calculation of population-weighted Gini indices and the initial machine learning attribution of equity drivers.

03_Thermal_Data.py: Spatial fusion of Landsat-derived Land Surface Temperature (LST) and VIIRS Nighttime Lights into neighborhood grids to validate the "Thermal Injustice" inverse law.

04_scenario_simulations.py: Large-scale scenario modeling to test the "Urban Acupuncture" dividend, contrasting targeted micro-greenery against traditional fringe development.

🛠 System Requirements

Language: Python 3.9 or higher.

Core Dependencies:

Spatial Analysis: geopandas, osmnx, rasterio, rasterstats

Machine Learning: scikit-learn, shap

Data Processing: pandas, numpy, tqdm

Visualization: matplotlib, seaborn

🚀 Instructions for Use

Accessibility Mapping: Run 01_Accessibility_2SFCA.py to calculate neighborhood-level green space access.

Equity Assessment: Run 02_Equity_and_ml_attribution.py to derive city-level metrics (generates Global_118_Cities_Equity_Metrics.csv).

Thermal Fusion: Execute 03_Thermal_Data.py to integrate heat risk and economic intensity into the spatial grids (generates Global_Micro_Grid_Thermal_Data.csv).

Intervention Simulation: Run 04_scenario_simulations.py to reproduce the global intervention trajectories and the health dividend analysis (generates All_118_Cities_Simulation_Results.csv).

📊 Data Sources

We utilize a multi-source data fusion approach:

Green Infrastructure: Polygons and network data from OpenStreetMap.

Vegetation Quality: Sentinel-2 NDVI derived via Google Earth Engine.

Thermal Data: Landsat-8/9 Surface Temperature (LST) products.

Demographics: WorldPop 2020 gridded population (100m resolution).

Economic Intensity: VIIRS Nighttime Lights (NTL) annual composites.

✉️ Contact

For questions regarding the code or datasets, please open an issue in this repository or contact the corresponding author.
