import pandas as pd
import numpy as np
import geopandas as gpd
import os
from pathlib import Path
from shapely import wkt
import re
import math


##########################################
## SET PATHS
##########################################

BASE_DIR = Path(os.getcwd()).parent
OSM_DIR = os.path.join(BASE_DIR, 'src', 'data', 'osm_features.csv')
ELE_DIR = os.path.join(BASE_DIR, 'src', 'data', 'elevation.csv')



##########################################
## LOADING DATA
##########################################

osm = pd.read_csv(OSM_DIR)
ele = pd.read_csv(ELE_DIR)

# Merge the two dataframes
features = pd.merge(osm, ele, on=['u', 'v', 'key'], how='left')

# Transform to GeoDataFrame
def transform(data):
    data = gpd.GeoDataFrame(data).set_index(["u", "v", "key"])
    data["geometry"] = data["geometry"].astype(str).apply(lambda x: wkt.loads(x))
    data = data.set_geometry('geometry')
    return data
features = transform(features)


##########################################
## OSM FEATURES SCORING FUNCTIONS
##########################################

# Function to calculate the raw scores from extracted features
def calculate_feature_score(row):
    osm_score = 0
    if row['bicycle_parking'] == 1:
        osm_score += 1
    if row['bus_stop'] == 0:
        osm_score += 1

    return osm_score/2


# Function to map road type to score
def road_type_to_score(road_type):
    if re.search(r'cycleway', road_type):
        return 1
    elif re.search(r'trunk', road_type) or re.search(r'motorway', road_type) or re.search(r'primary', road_type):
        return 0
    elif re.search(r'residential', road_type) or re.search(r'living_street', road_type):
        return 0.7
    elif re.search(r'pedestrian', road_type) or re.search(r'track', road_type):
        return 0.8
    elif road_type == 'path':
        return 0.7
    elif re.search(r'service', road_type):
        return 0.5
    elif re.search(r'secondary', road_type):
        return 0
    elif re.search(r'tertiary', road_type):
        return 0.2
    elif re.search(r'unclassified', road_type):
        return 0.6
    elif road_type == 'bridleway' or road_type == 'busway':
        return 0.5
    else:
        return np.nan
    

# Function to map pavement type to score
def pavement_type_to_score(surface):
    if re.search(r'asphalt|paved|concrete|concrete:plates', surface):
        return 1
    elif re.search(r'paving_stones|sett|fine_gravel|compacted|gravel', surface):
        return 0.8
    elif re.search(r'ground|clay|artificial_turf', surface):
        return 0.6
    elif re.search(r'grass|dirt|earth', surface):
        return 0.3
    elif re.search(r'cobblestone|unpaved', surface):
        return 0.1
    else:
        return np.nan
    

# Function to map width to score
def width_score(width):
    if width <= 10 and width > 0:
        return width / 10
    elif width > 10:
        return 1
    else:
        return np.nan



##########################################
## ELEVATION SCORING FUNCTION
##########################################

# Determine the maximum absolute elevation change
max_change = max(features['elevation_gain'].max(), abs(features['elevation_gain'].min()))

# Function to calculate the elevation gain score
def elevation_gain_score(elevation_gain, max_change=max_change):
    # Normalize the absolute elevation gain to a value between 0 and 1
    normalized_abs_gain = abs(elevation_gain) / max_change
    # Inverse score (flatter roads are preferred)
    score = 1 - normalized_abs_gain
    return score


##########################################
## CALCULATING SCORES
##########################################    
    
# Calculate osm scores
features['featureScore'] = features.apply(calculate_feature_score, axis=1)
features['roadTypeScore'] = features['highway'].astype(str).apply(road_type_to_score)
features['pavementTypeScore'] = features['pavement'].astype(str).apply(pavement_type_to_score)
features['widthScore'] = features['width'].apply(width_score)

# Calculate elevation scores
features['elevationGainScore'] = features['elevation_gain'].apply(elevation_gain_score)

# Calculate length scores (normalized between 0 and 1), shorter routes are better
features['lengthScore'] = 1 - (features['length'] / features['length'].max())


# Define the weights for each factor based on their importance
BASELINE_WEIGHTS = {
    'featureScore': 5,
    'roadTypeScore': 20,
    'pavementTypeScore': 25,
    'widthScore': 5,
    'elevationGainScore': 25,
    'lengthScore': 20
}

NOELEVATION_WEIGHTS = {
    'featureScore': 10,
    'roadTypeScore': 25,
    'pavementTypeScore': 30,
    'widthScore': 10,
    'elevationGainScore': 0,
    'lengthScore': 25
}

# Update the final score calculation to include weights
def calculate_weighted_final_score(row, WEIGHTS):
    # Calculate the weighted score for each factor, handling NaN values
    feature_score = row['featureScore'] * WEIGHTS['featureScore'] if not pd.isna(row['featureScore']) else 0
    type_score = row['roadTypeScore'] * WEIGHTS['roadTypeScore'] if not pd.isna(row['roadTypeScore']) else 0
    pavement_score = row['pavementTypeScore'] * WEIGHTS['pavementTypeScore'] if not pd.isna(row['pavementTypeScore']) else 0
    width_score = row['widthScore'] * WEIGHTS['widthScore'] if not pd.isna(row['widthScore']) else 0
    elevation_score = row['elevationGainScore'] * WEIGHTS['elevationGainScore'] if not pd.isna(row['elevationGainScore']) else 0
    length_score = row['lengthScore'] * WEIGHTS['lengthScore'] if not pd.isna(row['lengthScore']) else 0

    scores = [feature_score, type_score, pavement_score, width_score, elevation_score, length_score]
    
    # Calculate the sum of the weights for valid (non-NaN) scores
    valid_weights_sum = sum(WEIGHTS[key] for key, value in row.items() if key in WEIGHTS and not pd.isna(value))
    
    # Normalize the total score by the sum of valid weights
    total_score = sum(scores)
    if valid_weights_sum > 0:
        normalized_score = total_score / valid_weights_sum
    else:
        normalized_score = np.nan

    return normalized_score


# Apply the weighted final score calculation
features['baselineScore'] = features.apply(lambda row: 
    calculate_weighted_final_score(row, BASELINE_WEIGHTS), axis=1)

features['noelevationScore'] = features.apply(lambda row: 
    calculate_weighted_final_score(row, NOELEVATION_WEIGHTS), axis=1)

# Create reversed scores since osmnx MINIMIZES (instead of maximizing) on the weight parameter
# "Better" roads need to have lower scores
# max_val = math.ceil(features['weightedFinalScore'].max())
features['baselineScore_reversed'] = 1 - features['baselineScore']
features['noelevationScore_reversed'] = 1 - features['noelevationScore']



##########################################
## REMOVING HIGHWAYS
##########################################

# Remove highways from df
def contains_excluded_road_type(road_type):
    # List of road types to be excluded
    excluded_types = ['trunk', 'trunk_link', 'motorway', 'motorway_link', 'primary', 'secondary']
    # If the road_type is a string, check if it contains any excluded type
    if isinstance(road_type, str):
        return any(excluded in road_type for excluded in excluded_types)
    
    # If the road_type is a list, check if any element of the list is an excluded type
    elif isinstance(road_type, list):
        return any(any(excluded in item for excluded in excluded_types) for item in road_type)
    
    # Return False for other data types
    return False

# Apply the function to each element in the 'highway' column and filter out the matches
no_highway = features[~features['highway'].apply(contains_excluded_road_type)]



##########################################
## SAVING DATA
##########################################

OUT_DIR = os.path.join(BASE_DIR, 'models', 'scores_no_highways.csv')

no_highway['geometry'] = no_highway['geometry'].astype(str).apply(wkt.loads)
no_highway.to_csv(OUT_DIR, index=True)

