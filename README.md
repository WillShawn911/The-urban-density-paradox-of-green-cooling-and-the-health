# Global disparities in urban green space equity (or your actual paper title)

## Overview
This repository contains the data and Python code used for the spatial analysis, machine learning attribution, and scenario simulations in our study across 118 global cities.

## Repository Structure
- `/data`: Contains processed CSV files including city-level Gini coefficients, grid-level accessibility scores, and scenario simulation outputs. 
  *(Note: Raw satellite imagery from Sentinel-2/Landsat and raw WorldPop rasters are not included due to size limits, but can be freely downloaded from Google Earth Engine and their respective official portals).*
- `/scripts`: Contains Python scripts (.py) and Jupyter Notebooks (.ipynb) used for the analysis.

## System Requirements
- Python 3.9 or 3.10
- Required packages: `geopandas`, `pandas`, `numpy`, `osmnx`, `scikit-learn`, `shap`, `matplotlib`, `seaborn`.

## Instructions for Use
1. Run `01_data_preprocessing.py` to clean the raw OpenStreetMap polygons.
2. Run `02_modified_2SFCA_calculation.py` to generate the accessibility scores.
3. Run `04_random_forest_shap.py` to reproduce the machine learning driver attribution and SHAP summary plots.
