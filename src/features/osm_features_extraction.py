import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import MultiPolygon
from shapely import wkt

# Extract the graph from OSM
G = ox.graph_from_place('Stuttgart', network_type='bike')
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Reset the index of edges --> this is the gdf we are going to work with
edges_reset = edges.reset_index()
edges_reset['index'] = range(len(edges_reset))

# Function to extract features
def osm_features(city):
    # Get spots with bicycle parking
    bicycle_parking = ox.features_from_place(city, tags={"amenity": "bicycle_parking"}).reset_index(
        )[['amenity', 'geometry']].rename(columns={'amenity': 'bicycleParking'})
    # Get bus stops
    bus_stop = ox.features_from_place(city, tags={"highway": 'bus_stop'}).reset_index(
        )[['highway', 'geometry']].rename(columns={'highway': 'busStop'})
    # Pavement Type
    pavement = ox.features_from_place('Stuttgart', tags={'surface': True}).reset_index(
        )[['geometry','surface']].rename(columns={'surface': 'pavement'})
    # Remove MultiPolygons
    pavement = pavement[pavement['geometry'].apply(lambda x: not isinstance(x, MultiPolygon))]
    # Extracting width from edges
    width = edges_reset[['geometry', 'width']]
    
    return bicycle_parking, bus_stop, pavement, width


bicycle_parking, bus_stop, pavement, width = osm_features('Stuttgart')



##########################################
## MERGING OSM FEATURES BASED ON PROXIMITY TO EDGES
##########################################
# Dealing first with yes/no features

# Project the GeoDataFrame to a UTM CRS for more precise distance calculations
edges_projected = edges_reset.to_crs(epsg=32633)

# Adding new columns for OSM features in 'edges_projected'
edges_projected['bicycle_parking'] = 0 
edges_projected['bus_stop'] = 0

# List of features to loop through
features = [('bicycle_parking', bicycle_parking),
            ('bus_stop', bus_stop)]


# Function to perform spatial join between edges and features
def proximity(feature, buffer_radius=100):
    # Project the feature GeoDataFrame to the same CRS as the edges
    projected = feature.to_crs(epsg=32633)
    # Create a buffer around each feature
    projected['buffer'] = projected.buffer(buffer_radius)
    buffer = gpd.GeoDataFrame(geometry=projected['buffer'], crs=projected.crs)
    
    # Perform spatial join between buffers and edges
    # This finds edges that intersect with each feature's buffer
    nearest = gpd.sjoin(edges_projected, buffer, how='inner', predicate='intersects')
    
    return nearest
    

for feature_name, feature_gdf in features:
    # Applying spatial join function
    nearest = proximity(feature_gdf, buffer_radius=100)
    # Extract unique indices of edges that are near the feature
    indices = nearest['index'].unique()
    # Update the corresponding column in 'edges_projected' based on the feature
    edges_projected[feature_name] = edges_projected.index.map(lambda x: 1 if x in indices 
                                                              else edges_projected.loc[x, feature_name])



##########################################
## ADDING PAVEMENT TYPE
##########################################

# Applying proximity analysis as above
# Set bigger radius buffer --> minizing na values
nearest = proximity(pavement, buffer_radius=200)
temp = nearest[['index', 'index_right']]

pavement['index_p'] = range(len(pavement))
temp = temp.merge(pavement[['index_p', 'pavement']], left_on='index_right', 
                  right_on='index_p', how='left').drop(columns=['index_p'])

# Check most common pavement type for duplicate indices
temp = temp.groupby('index')['pavement'].agg(pd.Series.mode).reset_index()

# Handling arrays of pavement types --> taking the first value
# Function to extract the first element
def extraction(x):
    if isinstance(x, np.ndarray):
        return x[0] if x.size > 1 else x
    return x

temp['pavement'] = temp['pavement'].apply(extraction)

# Merge
edges_projected = edges_projected.merge(temp, on='index', how='left')



############################################
## HANDLING NA IN WIDTH COLUMN
############################################

# Adding index column to facilitate merging
width['index_w'] = range(len(width))

# Dealing with str values & lists of widths --> taking max value
def get_max(x):
    if isinstance(x, list):
        numeric_values = pd.to_numeric(pd.Series(x), errors='coerce')
        if not numeric_values.empty:
            return numeric_values.max()
        else:
            return np.nan
    else:
        return pd.to_numeric(x, errors='coerce')

width['width'] = width['width'].apply(get_max)

# Applying proximity analysis as above
# Set bigger radius buffer
nearest = proximity(width, buffer_radius=200)
temp = nearest[['index', 'index_right']]
temp = temp.merge(width[['index_w', 'width']], left_on='index_right', 
                  right_on='index_w', how='left').drop(columns=['index_w'])

# Calculating mean for duplicate indices
temp = temp.groupby('index')['width'].mean().reset_index()
temp['width'] = temp['width'].round(1)

# Merge; removing the old width column
edges_projected = edges_projected.drop(columns='width')
edges_projected = edges_projected.merge(temp, on='index', how='left')



############################################
## SAVE TO CSV
############################################

# Transforming back to WGS84
edges_reset = edges_projected.to_crs(epsg=4326)
# Remove index column
edges_reset = edges_reset.drop(columns='index')
    
# Reset the index
edges_reset = edges_reset.set_index(['u', 'v', 'key'])

# Saving edges with osm features as csv
BASE_DIR = Path(os.getcwd()).parent.parent
OUT_DIR = os.path.join(BASE_DIR, 'src', 'data', 'osm_features.csv')
edges_reset['geometry'] = edges_reset['geometry'].astype(str).apply(wkt.loads)
edges_reset.to_csv(OUT_DIR, index=True)


