import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint , Polygon, MultiPolygon

import requests
import time
from PIL import Image
from io import BytesIO
import random

# Fetch the street network from OSM
place_name = "Tübingen, Germany"
G = ox.graph_from_place(place_name, network_type='bike', simplify=True)

# Look at the network
#fig, ax = ox.plot_graph(ox.project_graph(G))

# Convert the graph into two GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Extract midpoints for every edge of the street network
edges['midpoint'] = ''
for i in range(0,len(edges)):
    midpoint = edges['geometry'].iloc[i].interpolate(0.5, normalized = True)
    edges['midpoint'].iloc[i] = midpoint
    

# TEST midpoints graph
#fig, ax = ox.plot_graph(G, node_size=5, figsize=(48, 48),node_color='r', show=False, close=False)    
#for i in range(0,len(edges)):
#    ax.scatter(edges['midpoint'].iloc[i].x, edges['midpoint'].iloc[i].y, c='g', s = 5)    
#plt.show() 
# END TEST midpoints graph


#### OPTION 1: Get streetview images for all roads (not practical for large cities)

# Be vary of the order of lat and lon!
""" What is the order of latitude and longitude coordinates?
In web mapping APIs like Google Maps, spatial coordinates are often in order of latitude then longitude.
In spatial databases like PostGIS and SQL Server, spatial coordinates are in longitude and then latitude.
"""

# Source API key from hidden file
google_api_key = "TEST_API_KEY"

def construct_streetview_url(lat, lon, api_key):
    """Construct a URL for the Street View Static API"""
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = f"?size=800x400&location={lat},{lon}&key={api_key}"
    return base_url + params

def is_image_empty(img):
    """Check if the image is one of those 'no image available' placeholders"""
    # Placeholder images often have very low color variance, this is a naive way to detect them
    return img.convert('L').nunique() < 10  # threshold, may need adjusting

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

def construct_filename(u, v, key, place_name=place_name):
    """Construct a filename based on edge identifiers."""
    return f"streetview_images/{place_name.split(',')[0]}/edge_{u}_{v}_{key}.jpg"

# Loop through the midpoints and retrieve the images

rate_limit_interval = 2  # seconds between requests

for edge_data in edges.itertuples():
    
    # Extract edge identifiers from the index
    u, v, key = edge_data.Index

    # Extract latitude and longitude from the midpoint
    lon = edge_data.midpoint.x
    lat = edge_data.midpoint.y

    # Construct URL and filename
    url = construct_streetview_url(lat, lon, google_api_key)
    filename = construct_filename(u, v, key)

    # Retrieve and save the image
    if retrieve_and_save_streetview_image(url, filename):
        print(f"Saved image for edge {u}-{v}-{key}")
    else:
        print(f"No image available for edge {u}-{v}-{key}")

    time.sleep(rate_limit_interval)  # rate limiting
    

#### OPTION 2: Too many roads > Cluster roads > Get midpoints of clusters > Get streetview images

from sklearn.cluster import DBSCAN

# Convert road midpoints to a numpy array for clustering
coords = np.array(edges['midpoint'])

# Extract longitude and latitude from each Point
lon_lat_array = np.array([(point.x, point.y) for point in edges['midpoint']])

# Convert longitude and latitude to radians
coords_in_radians = np.radians(lon_lat_array)

# Perform DBSCAN clustering
epsilon = 0.00001  # Cluster radius in degrees
min_samples = 1  # Minimum samples in a cluster
db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='auto', metric='haversine').fit(coords_in_radians)
edges['cluster'] = db.labels_

# Check the number of clusters formed
num_clusters = len(set(edges['cluster'])) - (1 if -1 in edges['cluster'] else 0)
print(f"Number of clusters: {num_clusters}")


## VISUALIZE CLUSTERS 1
import matplotlib.cm as cm

# Assign a color to each cluster using a colormap
num_clusters = len(set(edges['cluster'])) - (1 if -1 in edges['cluster'] else 0)
colormap = cm.get_cmap('viridis', num_clusters)

# Normalize cluster labels for color mapping
norm = plt.Normalize(vmin=edges['cluster'].min(), vmax=edges['cluster'].max())

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
edges.plot(column='cluster', ax=ax, cmap=colormap, norm=norm)

# Add color bar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, orientation='vertical')

plt.show()


## VISUALIZE CLUSTERS 2: VISUALIZATION WITH CONVEX HULLS

# Initialize a GeoDataFrame to store the convex hulls
hulls = gpd.GeoDataFrame(columns=['geometry', 'cluster'], crs=edges.crs)

# Calculate convex hull for each cluster
for cluster_id in set(edges['cluster']):
    if cluster_id != -1:  # Exclude noise points, if any
        # Extract points in this cluster
        points = [point for point, cluster in zip(edges['midpoint'], edges['cluster']) if cluster == cluster_id]
        
        # Create a convex hull
        if points:
            hull = MultiPoint(points).convex_hull
            hulls = hulls.append({'geometry': hull, 'cluster': cluster_id}, ignore_index=True)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
edges.plot(ax=ax, color='grey', alpha=0.5)  # Plot the edges in grey for context
hulls.plot(ax=ax, column='cluster', cmap=colormap, norm=norm, edgecolor='black')

plt.show()


### IMAGE RETRIEVAL FOR CLUSTERS

# Use the functions defined earlier to retrieve images for each cluster
# Additionally, we need to choose a representative point for each cluster to retrieve the image for

# Dictionary to store a representative point for each cluster
representative_points = {}

# Iterate through each cluster
for cluster_id in set(edges['cluster']):
    if cluster_id != -1:  # Exclude noise points, if any
        # Get all edges in this cluster
        cluster_edges = edges[edges['cluster'] == cluster_id]

        # Randomly select an edge
        selected_edge = cluster_edges.sample(n=1).iloc[0]

        # Store the midpoint of the selected edge
        representative_points[cluster_id] = selected_edge['midpoint']

# Now, representative_points contains a randomly selected midpoint for each cluster

def construct_filename_for_cluster(cluster_id):
    """Construct a filename based on the cluster ID"""
    return f"streetview_images/{place_name.split(',')[0]}/cluster_{cluster_id}.jpg"

# Loop through the representative points of each cluster and retrieve the images
for cluster_id, midpoint in representative_points.items():
    lon = midpoint.x
    lat = midpoint.y

    # Construct URL and filename for the cluster
    url = construct_streetview_url(lat, lon, google_api_key)
    filename = construct_filename_for_cluster(cluster_id)

    # Retrieve and save the image for the cluster
    if retrieve_and_save_streetview_image(url, filename):
        print(f"Saved image for cluster {cluster_id}")
    else:
        print(f"No image available for cluster {cluster_id}")

    time.sleep(rate_limit_interval)  # rate limiting
