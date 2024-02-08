import osmnx as ox
import pandas as pd
import numpy as np
import os
from pathlib import Path

from shapely.geometry import LineString, Polygon, MultiPolygon
from scipy.spatial import cKDTree

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
    
# Reset the index
edges_reset = edges_reset.set_index(['u', 'v', 'key'])


# Saving edges with osm features as csv
BASE_DIR = Path(os.getcwd()).parent.parent
OUT_DIR = os.path.join(BASE_DIR, 'src', 'data', 'osm_features.csv')
edges_reset['geometry'] = edges_reset['geometry'].astype(str).apply(wkt.loads)
edges_reset.to_csv(OUT_DIR, index=True)



# # Display all rows
# pd.set_option('display.max_rows', None)
# # Reset
# pd.reset_option('display.max_rows')