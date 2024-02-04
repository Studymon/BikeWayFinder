import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from scipy.spatial import cKDTree

import re
from shapely import wkt

# Extract the graph from OSM
G = ox.graph_from_place('Stuttgart', network_type='bike')
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Reset the index of edges --> this is the gdf we are going to work with
edges_reset = edges.reset_index()
edges_reset['index'] = range(len(edges_reset))

# Function to extract features
def osm_features(city):
    # Get nodes with the highway=traffic_signals tag (intersections with traffic lights)
    traffic_nodes = ox.features_from_place(city, tags={"highway": "traffic_signals"}).reset_index(
        )[['highway', 'geometry']].rename(columns={'highway': 'trafficSignals'})

    # Get spots with bicycle parking
    bicycle_parking = ox.features_from_place(city, tags={"amenity": "bicycle_parking"}).reset_index(
        )[['amenity', 'geometry']].rename(columns={'amenity': 'bicycleParking'})

    # Public transit options
    # Get tram stops
    transit_tram = ox.features_from_place(city, tags={"railway": 'tram_stop'}).reset_index(
        )[['railway', 'geometry']].rename(columns={'railway': 'tramStop'})
    # Get bus stops
    transit_bus = ox.features_from_place(city, tags={"highway": 'bus_stop'}).reset_index(
        )[['highway', 'geometry']].rename(columns={'highway': 'busStop'})

    # Get lighting
    lighting = ox.features_from_place(city, tags={'highway': 'street_lamp'}).reset_index(
        )[['highway', 'geometry']].rename(columns={'highway': 'lighting'})
    
    # On street parking
    street_parking_right = ox.features_from_place(city, tags={"parking:right": True}).reset_index(
        )[['geometry','parking:right']]
    street_parking_left = ox.features_from_place(city, tags={"parking:left": True}).reset_index(
        )[['geometry','parking:left']]
    street_parking_both = ox.features_from_place(city, tags={"parking:both": True}).reset_index(
        )[['geometry','parking:both']]
    
    # Pavement Type
    pavement = ox.features_from_place('Stuttgart', tags={'surface': True}).reset_index(
        )[['geometry','surface']].rename(columns={'surface': 'pavement'})
    # Remove MultiPolygons
    pavement = pavement[pavement['geometry'].apply(lambda x: not isinstance(x, MultiPolygon))]
    
    return traffic_nodes, bicycle_parking, transit_tram, transit_bus, lighting, street_parking_right, street_parking_left, street_parking_both, pavement


traffic_nodes, bicycle_parking, transit_tram, transit_bus, lighting, street_parking_right, street_parking_left, street_parking_both, pavement = osm_features('Stuttgart')



##########################################
## NEAREST NEIGHBOR ANALYSIS
##########################################

# Convert linestring geometries from edges gdf to a KDTree
coords = np.array([(line.xy[0][0], line.xy[1][0]) for line in edges.geometry])
tree = cKDTree(coords)

# Function to find the nearest edge to a point of the extracted features
def nearest_edges_point(node): 
    # Extracting coordinates from the points
    points_coordinates = node.geometry.apply(lambda point: [point.x, point.y]).to_list()
    # Converting to a NumPy array
    points_array = np.array(points_coordinates)
    # Perform the query
    distances, idx = tree.query(points_array)

    return idx

# Function add the nearest edge index to the extracted features and perform the merge
def merge_nearest_edges(node, edges_reset=edges_reset):
    # Apply the conversions based on geometry type
    node['geometry'] = node['geometry'].apply(lambda geom: geom.interpolate(0.5, normalized=True) if isinstance(geom, LineString) else geom)
    node['geometry'] = node['geometry'].apply(lambda geom: geom.centroid if isinstance(geom, Polygon) else geom)

    # Add the nearest line index to the points GeoDataFrame
    node['nearest_idx'] = nearest_edges_point(node)
    node = node.drop('geometry', axis=1)
    # Use drop_duplicates to keep only the first occurrence of each unique value in 'nearest_idx'
    node = node.drop_duplicates(subset='nearest_idx')

    # Now perform the merge
    edges_reset = edges_reset.merge(node, right_on='nearest_idx', left_on='index', how='left').drop('nearest_idx', axis=1)

    return edges_reset

# Loop through the features and perform the merge
features = [traffic_nodes, bicycle_parking, transit_tram, transit_bus, lighting, 
            street_parking_right, street_parking_left, street_parking_both, pavement]

for feature in features:
    edges_reset = merge_nearest_edges(feature, edges_reset=edges_reset)



##########################################
## SCORE CALCULATION
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
    
    
# Calculate scores
edges_reset['featureScore'] = edges_reset.apply(calculate_feature_score, axis=1)
edges_reset['scaledFeatureScore'] = edges_reset['featureScore'] / 8
edges_reset['roadTypeScore'] = edges_reset['highway'].astype(str).apply(road_type_to_score)
edges_reset['pavementTypeScore'] = edges_reset['pavement'].astype(str).apply(pavement_type_to_score)
edges_reset['meanWidth'] = edges_reset['width'].apply(calculate_mean_width)
edges_reset['widthScore'] = edges_reset['meanWidth'].apply(width_score)

# Calculate final score (taking into account NaN values in typeScore and widthScore)
def calculate_final_score(row):
    scaled_score = row['scaledFeatureScore']
    type_score = row['roadTypeScore']
    pavement_score = row['pavementTypeScore']
    width_score = row['widthScore']

    scores = [scaled_score, type_score, pavement_score, width_score]
    valid_scores = [score for score in scores if not pd.isna(score)]

    if valid_scores:
        return sum(valid_scores) / len(valid_scores)
    else:
        return np.nan

 
edges_reset['finalScore'] = edges_reset.apply(calculate_final_score, axis=1)

# Create reversed scores since osmnx MINIMIZES (instead of maximizing) on the weight parameter
# "Better" roads need to have lower scores
edges_reset['finalScore_reversed'] = 1 - edges_reset['finalScore']

# Reset the index
edges_reset = edges_reset.set_index(['u', 'v', 'key'])

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
edges_rest = edges_reset[~edges_reset['highway'].apply(contains_excluded_road_type)]

# Saving as a csv
edges_reset['geometry'] = edges_reset['geometry'].astype(str).apply(wkt.loads)
edges_reset.to_csv('osm_with_scores.csv', index=True)
edges_rest.to_csv('osm_scores_no_highways.csv', index=True)




# # Display all rows
# pd.set_option('display.max_rows', None)
# # Reset
# pd.reset_option('display.max_rows')