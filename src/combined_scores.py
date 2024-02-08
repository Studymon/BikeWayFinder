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
    raw_score = 0
    if row['trafficSignals'] == 'traffic_signals':
        raw_score += 1
    if row['bicycleParking'] == 'bicycle_parking':
        raw_score += 1
    if pd.isna(row['tramStop']):
        raw_score += 1
    if pd.isna(row['busStop']):
        raw_score += 1
    if row['lighting'] == 'street_lamp':
        raw_score += 1
    if pd.isna(row['parking:right']) or row['parking:right'] == 'no':
        raw_score += 1
    if pd.isna(row['parking:left']) or row['parking:left'] == 'no':
        raw_score += 1
    if pd.isna(row['parking:both']) or row['parking:both'] == 'no':
        raw_score += 1

    return raw_score


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
        return 0.1
    elif re.search(r'tertiary', road_type):
        return 0.2
    elif road_type == 'unclassified':
        return 0.6
    elif road_type == 'bridleway' or road_type == 'busway':
        return 0.5
    else:
        return np.nan
    

# Function to map pavement type to score
def pavement_type_to_score(surface):
    if re.search(r'asphalt|paved|concrete|tartan', surface):
        return 1
    elif re.search(r'paving_stones|sett|fine_gravel|compacted|gravel|chipseal', surface):
        return 0.8
    elif re.search(r'cobblestone|unpaved|pebblestone|sand|mud', surface):
        return 0.2
    elif re.search(r'grass|dirt|woodchips|earth', surface):
        return 0.3
    elif re.search(r'metal|wood|clay|stone|mulch|rubble|ground', surface):
        return 0.4
    elif re.search(r'concrete:plates|grass_paver|metal_grid|acrylic|tiles|stepping_stones|park|.*:lanes', surface):
        return 0.6
    else:
        return np.nan
    

# Function to calculate the mean width
def calculate_mean_width(width):
    if isinstance(width, list):
        # Extract numeric values from the list and calculate the mean
        values = [float(re.search(r'-?\d+\.\d+', str(val)).group()) for val in width if re.search(r'-?\d+\.\d+', str(val))]
        if values:
            return np.mean(values)
    else:
        # Handle single numeric value or other cases
        return float(re.search(r'-?\d+\.\d+', str(width)).group()) if re.search(r'-?\d+\.\d+', str(width)) else np.nan
    
# Function to map width to score
def width_score(width):
    if width <= 10 and width > 0:
        return width / 10
    elif width > 10:
        return 1
    else:
        return None



##########################################
## ELEVATION SCORING FUNCTION
##########################################

def elevation_gain_score(elevation_gain, max_acceptable_gain=100):
    # If elevation gain is positive (uphill), normalize it to a value between 0 and 1
    if elevation_gain > 0:
        normalized_gain = min(elevation_gain / max_acceptable_gain, 1)
        score = 1 - normalized_gain
    else:
        score = 1  # Flat ground or downhill preferred
    return score



##########################################
## CALCULATING SCORES
##########################################    
    
# Calculate osm scores
features['featureScore'] = features.apply(calculate_feature_score, axis=1)
features['scaledFeatureScore'] = features['featureScore'] / 8
features['roadTypeScore'] = features['highway'].astype(str).apply(road_type_to_score)
features['pavementTypeScore'] = features['pavement'].astype(str).apply(pavement_type_to_score)
features['meanWidth'] = features['width'].apply(calculate_mean_width)
features['widthScore'] = features['meanWidth'].apply(width_score)

# Calculate elevation scores
features['elevationGainScore'] = features['elevation_gain'].apply(lambda x: 
    elevation_gain_score(x))

# Define the weights for each factor based on their importance
WEIGHTS = {
    'scaledFeatureScore': 5,
    'roadTypeScore': 40,
    'pavementTypeScore': 25,
    'widthScore': 5,
    'elevationGainScore': 25
}

# Update the final score calculation to include weights
def calculate_weighted_final_score(row):
    scaled_score = row['scaledFeatureScore'] * WEIGHTS['scaledFeatureScore']
    type_score = row['roadTypeScore'] * WEIGHTS['roadTypeScore']
    pavement_score = row['pavementTypeScore'] * WEIGHTS['pavementTypeScore']
    width_score = row['widthScore'] * WEIGHTS['widthScore']
    elevation_score = row['elevationGainScore'] * WEIGHTS['elevationGainScore']

    scores = [scaled_score, type_score, pavement_score, width_score, elevation_score]
    valid_scores = [score for score in scores if not pd.isna(score)]
    
    if valid_scores:
        return sum(valid_scores) / len(valid_scores)
    else:
        return np.nan

# Apply the weighted final score calculation
features['weightedFinalScore'] = features.apply(calculate_weighted_final_score, axis=1)

# Create reversed scores since osmnx MINIMIZES (instead of maximizing) on the weight parameter
# "Better" roads need to have lower scores
max_val = math.ceil(features['weightedFinalScore'].max())
features['weightedFinalScore_reversed'] = max_val - features['weightedFinalScore']



##########################################
## SAVING DATA
##########################################

OUT_DIR = os.path.join(BASE_DIR, 'models', 'baseline_scores.csv')

features['geometry'] = features['geometry'].astype(str).apply(wkt.loads)
features.to_csv(OUT_DIR, index=True)



##########################################
## REMOVING HIGHWAYS
##########################################

# Reset the index
features = features.set_index(['u', 'v', 'key'])

# Remove highways from df
def contains_excluded_road_type(road_type):
    # List of road types to be excluded
    excluded_types = ['trunk', 'trunk_link', 'motorway', 'motorway_link', 'primary', 'primary_link', 'secondary']
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

features['geometry'] = features['geometry'].astype(str).apply(wkt.loads)
features.to_csv('osm_with_scores.csv', index=True)
no_highway.to_csv('osm_scores_no_highways.csv', index=True)


pd.set_option('display.max_columns', None)