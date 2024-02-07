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

# Script should be deterministic, but we still set seeds for reproducibility
np.random.seed(0)
random.seed(0)


#########################
#### FETCH STREET NETWORK
#########################

# Fetch the street network from OSM
place_name = "Stuttgart, Germany"
G = ox.graph_from_place(place_name, network_type='bike', simplify=True)

# Look at the network
#fig, ax = ox.plot_graph(ox.project_graph(G))

# Convert the graph into two GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Remove edges not suitable for cycling
edges = edges[~edges['highway'].isin(['primary', 'motorway', 'motorway_link', 'trunk', 'trunk_link'])]

# Calculate the midpoint of each edge
edges['midpoint'] = edges['geometry'].apply(lambda x: x.interpolate(0.5, normalized=True))

# Verify no midpoints are null
print(f"Number of edges with null midpoint: {edges['midpoint'].isnull().sum()}")

# Look at midpoints
fig, ax = ox.plot_graph(G, node_size=3, figsize=(60, 60),node_color='r', show=False, close=False)    
for i in range(0,len(edges)):
    ax.scatter(edges['midpoint'].iloc[i].x, edges['midpoint'].iloc[i].y, c='g', s = 1.5)    
plt.show() 


#####################
#### CLUSTERING EDGES
#####################

# Idea: Too many roads > Cluster roads > Get midpoints of clusters > Get streetview images

# Extract the x and y coordinates of the midpoints
coordinates = np.array([(point.x, point.y) for point in edges['midpoint']])

## DBSCAN clustering
dbscan = DBSCAN(eps=0.0005, min_samples=2)  # 0.0005 degrees is ~50 meters
clusters = dbscan.fit_predict(coordinates)
cluster_labels = dbscan.labels_
num_clusters = len(set(cluster_labels))
print('Number of clusters: {}'.format(num_clusters))
print('Number of noise points: {}'.format(np.sum(cluster_labels == -1)))


## Identify Cluster Centroids and Nearest Midpoints

# Add cluster labels to the edges DataFrame
edges['DBSCAN_group'] = clusters

# Group by cluster label
clustered = edges.groupby('DBSCAN_group')

# DataFrame to store centroids and nearest points
centroids_nearest = pd.DataFrame(columns=['DBSCAN_group', 'centroid', 'representative_point', 'linestring'])

for cluster_label, points in clustered:
    if cluster_label == -1:
        # Handle outliers separately
        continue

    # Calculate the centroid of the cluster and the nearest point in the cluster to the centroid
    multipoint = MultiPoint([point for point in points['midpoint']])
    centroid = multipoint.centroid
    representative_point = nearest_points(centroid, multipoint)[1]

    # Find the edge that corresponds to this representative_point and extract the linestring
    corresponding_edge = edges[edges['midpoint'] == representative_point].iloc[0]
    linestring = corresponding_edge.geometry

    centroids_nearest = centroids_nearest.append({'DBSCAN_group': cluster_label,
                                                  'centroid': centroid,
                                                  'representative_point': representative_point,
                                                  'linestring': linestring}, ignore_index=True)

# Handle outliers
outliers = edges[edges['DBSCAN_group'] == -1]
for idx, outlier in outliers.iterrows():
    centroids_nearest = centroids_nearest.append({'DBSCAN_group': -1,
                                                  'centroid': outlier['midpoint'],
                                                  'representative_point': outlier['midpoint'],
                                                  'linestring': outlier['geometry']}, ignore_index=True)

## Assign unique identifiers for each cluster and outlier
## This way the outliers are treated as separate clusters
## which we need as unique identifiers for SVI retrieval
cluster_id = 0
for idx, row in centroids_nearest.iterrows():
    centroids_nearest.at[idx, 'cluster_id'] = cluster_id
    cluster_id += 1
centroids_nearest['cluster_id'] = centroids_nearest['cluster_id'].astype(int)

# Merge to assign cluster_id to representative edges
edges = edges.merge(centroids_nearest[['representative_point', 'cluster_id']], left_on='midpoint', right_on='representative_point', how='left')

# Propagate cluster_id to all edges within the same DBSCAN group
for dbscan_group in centroids_nearest['DBSCAN_group'].unique():
    if dbscan_group == -1:
        # Outliers already have a unique cluster_id, skip them
        continue

    # Find the cluster_id assigned to the representative edge of this DBSCAN group
    cluster_id = centroids_nearest[centroids_nearest['DBSCAN_group'] == dbscan_group]['cluster_id'].iloc[0]

    # Assign this cluster_id to all edges in the same DBSCAN group
    edges.loc[edges['DBSCAN_group'] == dbscan_group, 'cluster_id'] = cluster_id

# Ensure cluster_id is an integer
edges['cluster_id'] = edges['cluster_id'].astype(int)

# Check how many clusters we have and how many of them are outliers
print(f"Number of clusters: {len(set(edges['cluster_id']))}")
print(f"Number of outliers: {len(edges[edges['DBSCAN_group'] == -1])}")


## Export the edges with clusters and centroids df to pickle files
os.chdir('c:\\Users\\andre\\iCloudDrive\\Dokumente\\Master Data Science\\3. Semester (WS 23)\\DS500-Data_Science_Project\\BikeWayFinder\\data')

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
fig, ax = ox.plot_graph(G, node_size=2, figsize=(60, 60), node_color='r', show=False, close=False) 

# Extract representative midpoints
representative_midpoints = [row['representative_point'] for idx, row in centroids_nearest.iterrows()]

# Plot each representative midpoint in green
for point in representative_midpoints:
    ax.scatter(point.x, point.y, c='green', s=2)

# Show the plot
plt.show()


#################################
#### Image Retrieval for Clusters
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

def construct_filename_for_cluster(cluster_id, place_name=place_name):
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

### Retrievel preparation

os.chdir('c:\\Users\\andre\\iCloudDrive\\Dokumente\\Master Data Science\\3. Semester (WS 23)\\DS500-Data_Science_Project\\BikeWayFinder\\data')
 
# Before we start, create the directory to store the images
os.makedirs(f"streetview_images/{place_name.split(',')[0]}/edges", exist_ok=True) 
  
# source the Google Streetview API key from separate hidden script
with open("./api_keys/svi_api_key.py") as script:
  exec(script.read())

rate_limit_interval = 0.05  # seconds between requests

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
    # Personally limited it to 20 requests per second or 1200 per minute, but:
    # Up to 30000 requests per minute allowed
    # Up to 25000 per day without digital signature
    # Costs: $7 for 1000 requests


image_retrieval_log = pd.DataFrame(image_retrieval_data)

log_file_path = "./log_files/edges_svi_retrieval_log.csv"
image_retrieval_log.to_csv(log_file_path, index=False)



########################
#### CLUSTERING OF NODES
########################

# Extract the x and y coordinates of the nodes
coordinates_n = np.array([(data['x'], data['y']) for node, data in nodes.to_dict('index').items()])

## DBSCAN clustering
dbscan_n = DBSCAN(eps=0.0006, min_samples=2)  # 0.0006 degrees is ~60 meters
clusters_n = dbscan_n.fit_predict(coordinates_n)
cluster_labels_n = dbscan_n.labels_
num_clusters_n = len(set(cluster_labels_n))
print('Number of clusters: {}'.format(num_clusters_n))
print('Number of noise points: {}'.format(np.sum(cluster_labels_n == -1)))

# Add cluster labels to the nodes DataFrame
nodes['DBSCAN_group'] = clusters_n

# Group by cluster label
clustered_nodes = nodes.groupby('DBSCAN_group')

# DataFrame to store centroids and nearest nodes
centroids_representative_node = pd.DataFrame(columns=['DBSCAN_group', 'centroid', 'representative_node'])


for cluster_label, points in clustered_nodes:
    if cluster_label == -1:
        # Handle outliers separately
        continue

    # Create a MultiPoint object from all nodes in the cluster
    multipoint = MultiPoint([Point(data['x'], data['y']) for node, data in points.to_dict('index').items()])
    # Calculate the centroid of the cluster
    centroid = multipoint.centroid
    # Find the nearest original node to the centroid
    representative_node = nearest_points(centroid, multipoint)[1]

    # Store the cluster label, centroid, and nearest node
    centroids_representative_node = centroids_representative_node.append({
        'DBSCAN_group': cluster_label,
        'centroid': centroid,
        'representative_node': representative_node
    }, ignore_index=True)


outliers = nodes[nodes['DBSCAN_group'] == -1]
for idx, outlier in outliers.iterrows():
    outlier_point = Point(outlier['x'], outlier['y'])
    centroids_representative_node = centroids_representative_node.append({
        'DBSCAN_group': -1,
        'centroid': outlier_point,
        'representative_node': outlier.name
    }, ignore_index=True)

# Assign unique identifiers for each cluster and outlier
cluster_id = 0
for idx, row in centroids_representative_node.iterrows():
    centroids_representative_node.at[idx, 'cluster_id'] = cluster_id
    cluster_id += 1
centroids_representative_node['cluster_id'] = centroids_representative_node['cluster_id'].astype(int)

# Add the cluster_id back to the original nodes DataFrame
nodes = nodes.merge(centroids_representative_node[['representative_node', 'cluster_id']], left_on=nodes.index, right_on='representative_node', how='left')

# Propagate cluster_id to all nodes within the same DBSCAN group
for dbscan_group in centroids_representative_node['DBSCAN_group'].unique():
    if dbscan_group == -1:
        # Outliers already have a unique cluster_id, skip them
        continue

    # Find the cluster_id assigned to the representative node of this DBSCAN group
    cluster_id = centroids_representative_node[centroids_representative_node['DBSCAN_group'] == dbscan_group]['cluster_id'].iloc[0]

    # Assign this cluster_id to all nodes in the same DBSCAN group
    nodes.loc[nodes['DBSCAN_group'] == dbscan_group, 'cluster_id'] = cluster_id
    
# check for null values in cluster_id
print(f"Number of nodes with null cluster_id: {nodes['cluster_id'].isnull().sum()}")

# Check how many observations we have of DBSCAN_group -1
print(f"Number of observations with DBSCAN_group -1: {len(nodes[nodes['DBSCAN_group'] == -1])}")

# Ensure cluster_id is an integer
nodes['cluster_id'] = nodes['cluster_id'].astype(int)

# Check how many clusters we have and how many of them are outliers
print(f"Number of clusters: {len(set(nodes['cluster_id']))}")


## Export the nodes with clusters and centroids df to pickle files
os.chdir('c:\\Users\\andre\\iCloudDrive\\Dokumente\\Master Data Science\\3. Semester (WS 23)\\DS500-Data_Science_Project\\BikeWayFinder\\data')

# Export `edges` to a pickle file
file_name_nodes  = f"nodes_{place_name.split(',')[0]}_clustered"
file_path_nodes  = os.path.join("interim", file_name_nodes)
file_path_nodes = f"{file_path_nodes}.pkl"
nodes.to_pickle(file_path_nodes)

# Export `centroids_representative_node` to a pickle file
file_name_centroids_n  = f"centroids_of_clustered_nodes_{place_name.split(',')[0]}"
file_path_centroids_n  = os.path.join("interim", file_name_centroids_n)
file_path_centroids_n = f"{file_path_centroids_n}.pkl"
centroids_representative_node.to_pickle(file_path_centroids_n)

## Import

# Import `edges` from the pickle file
nodes_imported = pd.read_pickle(file_path_nodes)
centroids_representative_node_imported = pd.read_pickle(file_path_centroids_n)


# Plot only the edges of the graph
#fig, ax = ox.plot_graph(G, node_size=0, figsize=(60, 60), show=False, close=False)

# Assuming centroids_representative_node['representative_node'] contains the representative Point objects
#for idx, row in centroids_representative_node.iterrows():
#    point = row['representative_node']
#    if point:  # Check if point is not None
#        ax.scatter(point.x, point.y, c='green', s=4, zorder=2)  # zorder=2 to plot on top of the network

# Show the plot
#plt.show()


##########################################
#### Image Retrieval for Clusters of Nodes
##########################################

os.chdir('c:\\Users\\andre\\iCloudDrive\\Dokumente\\Master Data Science\\3. Semester (WS 23)\\DS500-Data_Science_Project\\BikeWayFinder\\data')
 
# Before we start, create the directory to store the images
os.makedirs(f"streetview_images/{place_name.split(',')[0]}/nodes", exist_ok=True) 

# Re-defining construct_streetview_url function without heading
def construct_streetview_url(lat, lon, api_key, pitch=0, fov=90):
    """Construct a URL for the Street View Static API."""
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = f"?size=640x400&location={lat},{lon}&pitch={pitch}&fov={fov}&key={api_key}"
    # image size max is 640x640, squared doesn't look good
    return base_url + params

# Re-defining construct_filename_for_cluster function for nodes naming
def construct_filename_for_cluster(cluster_id, place_name=place_name):
    """Construct a filename based on the cluster ID"""
    return f"streetview_images/{place_name.split(',')[0]}/nodes/nodes-cluster_{cluster_id}.jpg"

rate_limit_interval = 0.05  # seconds between requests

# Log the status of image retrieval
image_retrieval_data_n = []

### Retrieve images for clusters
for idx, row in centroids_representative_node.iterrows():
    cluster_id = row['cluster_id']

    # Extract latitude and longitude from the representative point
    lon = row['representative_node'].x
    lat = row['representative_node'].y

    # Construct URL and filename
    url = construct_streetview_url(lat, lon, google_api_key, pitch=-10)
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
    image_retrieval_data_n.append({
        'cluster_id': cluster_id,
        'status': status,
        'filename': filename,
        'url': url
    })

    time.sleep(rate_limit_interval)
    # Notes on rate limiting:
    # Personally limited it to 20 requests per second or 1200 per minute, but:
    # Up to 30000 requests per minute allowed
    # Up to 25000 per day without digital signature
    # Costs: $7 for 1000 requests


image_retrieval_log_n = pd.DataFrame(image_retrieval_data_n)

log_file_path_n = "./log_files/nodes_svi_retrieval_log.csv"
image_retrieval_log_n.to_csv(log_file_path_n, index=False)

