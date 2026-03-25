"""
Phase 2: Modified 2-Step Floating Catchment Area (2SFCA)
This script calculates walking-distance green space accessibility 
at the neighborhood level, weighted by NDVI quality.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd

# Define paths
PROCESSED_DIR = "./data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 2SFCA Parameters
WALKING_SPEED_MPS = 1.25 # 1.25 meters per second (approx 4.5 km/h)
TIME_THRESHOLD_SEC = 900 # 15 minutes
MAX_CAP_SQM_PER_CAPITA = 10000 # Winsorization cap

def step1_supply_demand_ratio(parks_gdf, pop_grids, distance_matrix):
    """
    Step 1: Calculate supply-to-demand ratio (Rj) for each park.
    Rj = (Area * NDVI) / Total_Population_within_Catchment
    """
    r_ratios = {}
    for park_idx, park in parks_gdf.iterrows():
        # Find populations within 15-min walking distance
        reachable_pops = distance_matrix[(distance_matrix['park_id'] == park_idx) & 
                                         (distance_matrix['distance_m'] <= WALKING_SPEED_MPS * TIME_THRESHOLD_SEC)]
        
        total_pop_demand = reachable_pops['population'].sum()
        
        if total_pop_demand > 0:
            quality_weighted_supply = park['area_sqm'] * park['mean_ndvi']
            r_ratios[park_idx] = quality_weighted_supply / total_pop_demand
        else:
            r_ratios[park_idx] = 0
            
    return r_ratios

def step2_accessibility_score(pop_grids, r_ratios, distance_matrix):
    """
    Step 2: Calculate accessibility score (Ai) for each population grid.
    Ai = Sum of Rj for all reachable parks
    """
    accessibility_scores = []
    
    for grid_idx, grid in pop_grids.iterrows():
        # Find parks reachable from this grid within 15 mins
        reachable_parks = distance_matrix[(distance_matrix['grid_id'] == grid_idx) & 
                                          (distance_matrix['distance_m'] <= WALKING_SPEED_MPS * TIME_THRESHOLD_SEC)]
        
        total_score = sum([r_ratios.get(p_id, 0) for p_id in reachable_parks['park_id']])
        
        # Apply winsorization cap
        total_score = min(total_score, MAX_CAP_SQM_PER_CAPITA)
        accessibility_scores.append(total_score)
        
    pop_grids['accessibility_score'] = accessibility_scores
    return pop_grids

if __name__ == "__main__":
    print("Initiating 2SFCA calculations...")
    # This is a structural representation. In production, this loop iterates over 118 cities,
    # loading pre-calculated network distance matrices (OSMnx/NetworkX) and raster values.