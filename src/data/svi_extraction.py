import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint , Polygon, MultiPolygon
from shapely.ops import linemerge, nearest_points
from shapely import wkt
from shapely.wkt import loads, dumps
from sklearn.cluster import DBSCAN

import math
import random

import os
import requests
import time
from PIL import Image
from io import BytesIO

######################################
#### Image Feature Pipeline Part 1/5:
#### >> Clustering + SVI Extraction <<
#### Sorting SVI [notebook]
#### Image Segmentation [notebook]
#### Missing Cluster Handling
#### Image Feature Extraction
######################################

# Script should be deterministic, but we still set seeds for reproducibility
np.random.seed(0)
random.seed(0)


#################################
#### Functions For SVI Retrieval
#################################

# Be vary of the order of lat and lon!
""" What is the order of latitude and longitude coordinates?
In web mapping APIs like Google Maps, spatial coordinates are often in order of latitude then longitude.
In spatial databases like PostGIS and SQL Server, spatial coordinates are in longitude and then latitude.
"""

### Define functions to calculate heading, construct URL and filename,
### and retrieve and save the image

# Function to calculate heading between two points
def segment_heading(start, end):
    delta_lon = math.radians(end[0] - start[0])
    lat1, lat2 = map(math.radians, [start[1], end[1]])

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

    bearing = math.atan2(x, y)
    return math.degrees(bearing)

def calculate_heading(linestring):
    """Calculate heading based on a representative segment of the LINESTRING."""

    # Number of segments to consider for averaging the heading
    num_segments = min(5, len(linestring.coords) - 1)  # Adjust based on the length of linestring
    
    # Calculate the heading for each segment and average them
    total_heading = 0
    for i in range(num_segments):
        segment_start = linestring.coords[i]
        segment_end = linestring.coords[i + 1]
        total_heading += segment_heading(segment_start, segment_end)

    average_heading = total_heading / num_segments
    heading = (average_heading + 360) % 360  # Normalize to 0-360

    return heading

def construct_streetview_url(lat, lon, heading, api_key, pitch=0, fov=90):
    """Construct a URL for the Street View Static API."""
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = f"?size=640x400&location={lat},{lon}&heading={heading}&pitch={pitch}&fov={fov}&key={api_key}"
    # image size max is 640x640, squared doesn't look good
    return base_url + params

def is_image_empty(img):
    """Check if the image is one of those 'no image available' placeholders."""
    threshold = 20  # threshold for standard deviation of pixel values, determined by experimentation
    # Convert to grayscale and compute the standard deviation of pixel values
    stdev = np.std(np.array(img.convert('L')))
    return stdev < threshold

def construct_filename_for_cluster(cluster_id, place_name):
    """Construct a filename based on the cluster ID"""
    return f"streetview_images/{place_name.split(',')[0]}/edges/edges-cluster_{cluster_id}.jpg"

def retrieve_and_save_streetview_image(url, filename):
    """Retrieve a Street View image and save it to a file."""
    response = requests.get(url)
    if response.status_code == 200:
        # Check if the response contains an actual image
        img = Image.open(BytesIO(response.content))
        if not is_image_empty(img):
            img.save(filename)
            return True
    return False

#################################
## Function for Removing Highways
#################################

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


#########################
#### Fetch Street Network
#########################

# Fetch the street network from OSM
place_name = "Stuttgart, Germany"
G = ox.graph_from_place(place_name, network_type='bike', simplify=True)

# Get the nodes from the graph
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Change to the data directory
os.chdir('../../data')


""" Note on building the street network:
The clustering and therefore image retrieval for Stuttgart was based on an interims version of
`edges`, which can be recreated with the commented-out section A, using the GeoDataFrame
`osm_scores_no_highways.pkl` as edges.

Section B allows to cluster the edges and retrieve images for a city of choice, set with `place_name` above.
"""
###########
# Section A
###########
# Load the cleaned edges as GeoDataFrame from pickle file
#edges = pd.read_pickle('processed/osm_scores_no_highways.pkl')
# Create the network with the cleaned edges
#G = ox.graph_from_gdfs(nodes, edges)

###########
# Section B

## Clean the edges from highways
# Ensure correct index
edges = edges.set_index(['u', 'v', 'key'])
edges = edges[~edges['highway'].apply(contains_excluded_road_type)]

# Create the network with the cleaned edges
G = ox.graph_from_gdfs(nodes, edges)

# Section B ends here
###########


# Look at the network
#fig, ax = ox.plot_graph(G, node_size=3, figsize=(60, 60),node_color='r', show=False, close=False)    
#plt.show() 

# Calculate the midpoint of each edge
edges['midpoint'] = edges['geometry'].apply(lambda x: x.interpolate(0.5, normalized=True))

# Verify no midpoints are null
print(f"Number of edges with null midpoint: {edges['midpoint'].isnull().sum()}")

# Look at midpoints
#fig, ax = ox.plot_graph(G, node_size=3, figsize=(60, 60),node_color='r', show=False, close=False)    
#for i in range(0,len(edges)):
#    ax.scatter(edges['midpoint'].iloc[i].x, edges['midpoint'].iloc[i].y, c='g', s = 1.5)    
#plt.show() 


#####################
#### Clustering Edges
#####################

# Idea: Too many roads > Cluster roads > Get midpoints of clusters > Get streetview images

# Extract the x and y coordinates of the midpoints
coordinates = np.array([(point.x, point.y) for point in edges['midpoint']])

## DBSCAN clustering
dbscan = DBSCAN(eps=0.00030, min_samples=2)  # 0.0003 degrees is ~30 meters
clusters = dbscan.fit_predict(coordinates)
cluster_labels = dbscan.labels_
num_clusters = len(set(cluster_labels))
print('Number of clusters: {}'.format(num_clusters))
print('Number of noise points: {}'.format(np.sum(cluster_labels == -1)))


## Identify Cluster Centroids and Nearest Midpoints

# Add cluster labels to the edges DataFrame
edges['DBSCAN_group'] = clusters

# Store centroids and nearest points
data = []

for cluster_label, points in edges.groupby('DBSCAN_group'):
    if cluster_label == -1:
        # Handle outliers separately
        continue

    # Calculate the centroid of the cluster and the nearest point in the cluster to the centroid
    multipoint = MultiPoint(points['midpoint'].tolist())
    centroid = multipoint.centroid
    representative_point = nearest_points(centroid, multipoint)[1]

    # To find the corresponding edge, we match the 'midpoint' with 'representative_point'
    # This assumes that 'midpoint' is directly comparable with 'representative_point'
    corresponding_edge = points[points['midpoint'].apply(lambda x: x == representative_point)].iloc[0]
    linestring = corresponding_edge.geometry

    data.append({'DBSCAN_group': cluster_label,
                    'centroid': centroid,
                    'representative_point': representative_point,
                    'linestring': linestring})

# Handle outliers
outliers = edges[edges['DBSCAN_group'] == -1]
for idx, outlier in outliers.iterrows():
    data.append({'DBSCAN_group': -1,
                    'centroid': outlier['midpoint'],
                    'representative_point': outlier['midpoint'],
                    'linestring': outlier['geometry']})

centroids_nearest = pd.DataFrame(data)

## Assign unique identifiers for each cluster and outlier
## This way the outliers are treated as separate clusters
## which we need as unique identifiers for SVI retrieval
cluster_id = 0
for idx, row in centroids_nearest.iterrows():
    centroids_nearest.at[idx, 'cluster_id'] = cluster_id
    cluster_id += 1
centroids_nearest['cluster_id'] = centroids_nearest['cluster_id'].astype(int)

## Merge to assign cluster_id to representative edges
# Preserve the original multi-index by resetting it and keeping it as columns
edges_reset = edges.reset_index()
# Merge
edges = edges_reset.merge(centroids_nearest[['representative_point', 'cluster_id']],
                          left_on='midpoint', right_on='representative_point', how='left')

# Propagate cluster_id to all edges within the same DBSCAN group
edges['cluster_id'] = edges.groupby('DBSCAN_group')['cluster_id'].transform('first').astype(int)
edges.set_index(['u', 'v', 'key'], inplace=True)

# Check how many clusters we have and how many of them are outliers
print(f"Number of clusters: {len(set(edges['cluster_id']))}")
print(f"Number of outliers: {len(edges[edges['DBSCAN_group'] == -1])}")


## Export the edges with clusters and centroids df to pickle files
os.getcwd()
# Export `edges` to a pickle file
file_name_edges  = f"edges_{place_name.split(',')[0]}_clustered"
file_path_edges  = os.path.join("interim", file_name_edges)
file_path_edges = f"{file_path_edges}.pkl"
edges.to_pickle(file_path_edges)

# Export `centroids_nearest` to a pickle file
file_name_centroids  = f"centroids_of_clustered_edges_{place_name.split(',')[0]}"
file_path_centroids  = os.path.join("interim", file_name_centroids)
file_path_centroids = f"{file_path_centroids}.pkl"
centroids_nearest.to_pickle(file_path_centroids)

## Import

# Import `edges` from the pickle file
edges_imported = pd.read_pickle(file_path_edges)
centroids_nearest_imported = pd.read_pickle(file_path_centroids)


## VISUALIZE CLUSTER MIDPOINTS
#fig, ax = ox.plot_graph(G, node_size=2, figsize=(60, 60), node_color='r', show=False, close=False) 

# Extract representative midpoints
#representative_midpoints = [row['representative_point'] for idx, row in centroids_nearest.iterrows()]

# Plot each representative midpoint in green
#for point in representative_midpoints:
#    ax.scatter(point.x, point.y, c='green', s=2)

# Show the plot
#plt.show()


#################################
#### Image Retrieval for Clusters
#################################

### Retrievel preparation
os.getcwd()  # verify current working directory is data
 
# Before we start, create the directory to store the images
os.makedirs(f"streetview_images/{place_name.split(',')[0]}/edges", exist_ok=True) 
  
# source the Google Streetview API key from separate hidden script
with open("./api_keys/svi_api_key_sa.py") as script:
  exec(script.read())

rate_limit_interval = 0.01  # seconds between requests

# Log the status of image retrieval
image_retrieval_data = []

### Retrieve images for clusters
for idx, row in centroids_nearest.iterrows():
    cluster_id = row['cluster_id']
    linestring = row['linestring']

    # Calculate heading of image with linestring geometry of edge
    heading = calculate_heading(linestring)

    # Extract latitude and longitude from the representative point
    lon = row['representative_point'].x
    lat = row['representative_point'].y

    # Construct URL and filename
    url = construct_streetview_url(lat, lon, heading, google_api_key, pitch=-10)
    filename = construct_filename_for_cluster(cluster_id, place_name)
    
    # Initialize status for logging
    status = 'Image Not Available'

    # Retrieve and save the image
    if retrieve_and_save_streetview_image(url, filename):
        status = 'Image Saved'
        print(f"Saved image for cluster {cluster_id}")
    else:
        print(f"No image available for cluster {cluster_id}")
        
    # Log the attempt
    image_retrieval_data.append({
        'cluster_id': cluster_id,
        'status': status,
        'filename': filename,
        'url': url
    })

    time.sleep(rate_limit_interval)
    # Notes on rate limiting:
    # Personally limited it to 100 requests per second or 6000 per minute, but:
    # Up to 30000 requests per minute allowed
    # Up to 25000 per day without digital signature
    # Costs: $7 for 1000 requests


image_retrieval_log = pd.DataFrame(image_retrieval_data)

log_file_path = "./log_files/edges_svi_retrieval_log.csv"
image_retrieval_log.to_csv(log_file_path, index=False)
