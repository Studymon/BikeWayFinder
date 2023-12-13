import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np

import re
from shapely import wkt

# Function to extract geo data from OSM
def geodata_to_df(country, city):

    G = ox.graph_from_place(city, network_type='bike')  # download raw geospatial data from OSM

    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    nodes["city"], edges["city"] = city, city
    nodes["country"], edges["country"] = country, country

    edges["lat_long"] = edges["geometry"].apply(lambda x: re.sub(r'[^0-9., ]', "", str([re.sub(r'[^0-9. ]', '', str(i)) for i in list(zip(x.xy[1], x.xy[0]))])))
    edges["geometry"] = edges["geometry"].apply(lambda x: wkt.dumps(x))

    edges["highway"] = edges["highway"].apply(lambda x: ", ".join(x) if x.__class__.__name__=="list" else x)
    edges["name"] = edges["name"].apply(lambda x: ", ".join(x) if x.__class__.__name__=="list" else x)
    edges["maxspeed"] = edges["maxspeed"].apply(lambda x: ", ".join(x) if x.__class__.__name__ == "list" else x)
    edges["ref"] = edges["ref"].apply(lambda x: ", ".join(x) if x.__class__.__name__ == "list" else x)
    edges["reversed"] = edges["reversed"].apply(lambda x: x[0] if x.__class__.__name__ == "list" else x)
    edges["oneway"] = edges["oneway"].apply(lambda x: x[0] if x.__class__.__name__ == "list" else x)

    edges.fillna(-99, inplace=True)
    nodes.fillna(-99, inplace=True)
    edges["name"] = edges["name"].astype(str).replace("-99", None)

    # nodes_and_edges = gpd.sjoin(edges, nodes, how="left", predicate="intersects")

    return G, nodes, edges

G, nodes, edges = geodata_to_df('Germany', 'Stuttgart')

# Get unlisted osmid in edges (since some osmid contain listed values)
edges_unlist = edges.explode('osmid').reset_index()

# Function to extract OSM features
def osm_features(city):
    # Create a GeoDataFrame for intersections
    # intersections = gpd.GeoDataFrame(geometry=nodes.geometry)

    # Get nodes with the highway=traffic_signals tag (intersections with traffic lights)
    traffic_nodes = ox.features_from_place(city, tags={"highway": "traffic_signals"}).reset_index()[['osmid','highway']].rename(columns={'highway': 'trafficSignals'})

    # Get spots with bicycle parking
    bicycle_parking = ox.features_from_place(city, tags={"amenity": "bicycle_parking"}).reset_index()[['osmid','amenity']].rename(columns={'amenity': 'bicycleParking'})

    # Public transit options
    # Get tram stops
    transit_tram = ox.features_from_place(city, tags={"railway": 'tram_stop'}).reset_index()[['osmid','railway']].rename(columns={'railway': 'tramStop'})
    # Get bus stops
    transit_bus = ox.features_from_place(city, tags={"highway": 'bus_stop'}).reset_index()[['osmid','highway']].rename(columns={'highway': 'busStop'})

    # Get lighting
    lighting = ox.features_from_place(city, tags={'highway': 'street_lamp'}).reset_index()[['osmid','highway']].rename(columns={'highway': 'lighting'})
    
    # On street parking
    street_parking_right = ox.features_from_place(city, tags={"parking:right": True})['parking:right'].reset_index()[['osmid','parking:right']]
    street_parking_left = ox.features_from_place(city, tags={"parking:left": True})['parking:left'].reset_index()[['osmid','parking:left']]
    street_parking_both = ox.features_from_place(city, tags={"parking:both": True})['parking:both'].reset_index()[['osmid','parking:both']]
    
    # Merge all features
    geodfs_to_merge = [bicycle_parking, transit_tram, transit_bus, lighting,
                       street_parking_right, street_parking_left, street_parking_both]

    # Initial merge with nodes_and_edges
    merged_osm = traffic_nodes

    # Perform outer merges in a loop
    for geodf in geodfs_to_merge:
        merged_osm = merged_osm.merge(geodf, on='osmid', how='outer')
        
    return merged_osm


# Function to calculate the raw scores from extracted features
def calculate_raw_score(row):
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
    elif re.search(r'trunk', road_type):
        return 0
    elif re.search(r'service', road_type) or re.search(r'track', road_type):
        return 0.1
    elif re.search(r'primary', road_type):
        return 0.2
    elif re.search(r'secondary', road_type):
        return 0.4
    elif re.search(r'tertiary', road_type):
        return 0.5
    elif re.search(r'residential', road_type) or re.search(r'living_street', road_type):
        return 0.7
    elif re.search(r'pedestrian', road_type):
        return 0.8
    elif road_type == 'path':
        return 0.7
    elif road_type == 'unclassified':
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
    if width <= 10:
        return width / 10
    elif width > 10:
        return 1
    else:
        return None



### Appying above functions

merged_features = osm_features('Stuttgart')

# Merge with edges_unlist
merged_osm = edges_unlist.merge(merged_features, on='osmid', how='outer')

# Calculate scores
merged_osm['rawScore'] = merged_osm.apply(calculate_raw_score, axis=1)
merged_osm['scaledScore'] = merged_osm['rawScore'] / 8
merged_osm['typeScore'] = merged_osm['highway'].astype(str).apply(road_type_to_score)
merged_osm['meanWidth'] = merged_osm['width'].apply(calculate_mean_width)
merged_osm['widthScore'] = merged_osm['meanWidth'].apply(width_score)


# Calculate final score (taking into account NaN values in typeScore and widthScore)
def calculate_final_score(row):
    scaled_score = row['scaledScore']
    type_score = row['typeScore']
    width_score = row['widthScore']

    if pd.isna(type_score) and pd.isna(width_score):
        return scaled_score
    elif pd.isna(width_score):
        return (scaled_score + type_score) / 2
    elif pd.isna(type_score):
        return (scaled_score + width_score) / 2
    else:
        return (scaled_score + type_score + width_score) / 3
    
merged_osm['finalScore'] = merged_osm.apply(calculate_final_score, axis=1)









