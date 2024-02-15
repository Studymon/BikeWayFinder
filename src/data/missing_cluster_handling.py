import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

import os
import zipfile
import gzip
import io
import torch

#####################################
#### Image Feature Pipeline Part 4/5:
#### Clustering + SVI Extraction
#### Sorting SVI [notebook]
#### Image Segmentation [notebook]
#### >> Missing Cluster Handling <<
#### Image Feature Extraction
######################################

# Function for importing the compressed tensors containing image segmentation information
def load_compressed_tensor(filename):
    with gzip.open(filename + '.gz', 'rb') as f:
        buffer = f.read()
    tensor = torch.load(io.BytesIO(buffer))
    return tensor

################
#### Preparation
################

# Change to the data directory
os.chdir('../../data_visible')

# Unzip the folder containing the tensors (only need to do this once)
#zipped_folder_path = 'interim/edges_pred_output.zip'
#unzip_destination = 'interim'

#with zipfile.ZipFile(zipped_folder_path, 'r') as zip_ref:
#    zip_ref.extractall(unzip_destination)

place_name = "Stuttgart, Germany"

# Import the clustered edges and cluster information [see src/features/svi_extraction.py]
edges_clustered = pd.read_pickle(f"interim/edges_{place_name.split(',')[0]}_clustered.pkl")
nodes = pd.read_pickle(f"interim/nodes_{place_name.split(',')[0]}.pkl")  # for building the graph G later
centroids_nearest = pd.read_pickle(f"interim/centroids_of_clustered_edges_{place_name.split(',')[0]}.pkl")

# Drop unused columns
centroids_nearest = centroids_nearest.drop(columns=['DBSCAN_group', 'linestring'])


##################################################
#### Find closest cluster for all missing clusters
##################################################

# Import the filenames of the tensors
edges_pred_output_files = os.listdir(f'interim/svi/{place_name.split(',')[0]}/edges_pred_output')

# Extracting non-missing cluster IDs from filenames
retrieved_cluster_ids = [int(filename.split('_')[1].split('_')[0]) for filename in edges_pred_output_files]

# Identifying missing cluster IDs
all_cluster_ids = centroids_nearest['cluster_id'].tolist()
missing_cluster_ids = [cluster_id for cluster_id in all_cluster_ids if cluster_id not in retrieved_cluster_ids]

# Create a dataframe for clusters for which we have images
clusters_with_images = centroids_nearest[centroids_nearest['cluster_id'].isin(retrieved_cluster_ids)]

# Convert representative points of clusters with images into a format suitable for spatial distance calculations
representative_points = np.array(list(map(lambda x: (x.x, x.y), clusters_with_images['representative_point'])))

# Building a spatial index for efficient distance calculations
tree = cKDTree(representative_points)

# Dictionary to hold the closest cluster for each missing cluster
closest_clusters = {}

# For each missing cluster, find the closest cluster with an image
for missing_cluster_id in missing_cluster_ids:
    missing_centroid = centroids_nearest.loc[centroids_nearest['cluster_id'] == missing_cluster_id, 'centroid'].values[0]
    distance, index = tree.query((missing_centroid.x, missing_centroid.y))
    closest_cluster_id = clusters_with_images.iloc[index]['cluster_id']
    closest_clusters[missing_cluster_id] = closest_cluster_id


####################
## Sanity check 1/2: 
## Visualize missing clusters and their closest clusters on map
####################

# Create a dataframe for clusters for which we do not have images
clusters_without_images = centroids_nearest[centroids_nearest['cluster_id'].isin(missing_cluster_ids)]

# Add the closest cluster to the clusters without images
closest_clusters_df = pd.DataFrame({'cluster_id': list(closest_clusters.keys()), 
                                    'closest_cluster_id': list(closest_clusters.values())})
clusters_without_images = clusters_without_images.merge(closest_clusters_df, on='cluster_id', how='left')

# Add the represantative point of the closest cluster
closest_cluster_points = clusters_with_images[['cluster_id', 'representative_point']]
closest_cluster_points.columns = ['closest_cluster_id', 'closest_representative_point']
clusters_without_images = clusters_without_images.merge(closest_cluster_points, on='closest_cluster_id', how='left')


## VISUALIZATION
# Create the network with the cleaned edges
G = ox.graph_from_gdfs(nodes, edges_clustered)

# Extract representative midpoints
actual_cluster_points = [row['representative_point'] for idx, row in clusters_without_images.iterrows()]
closest_cluster_points = [row['closest_representative_point'] for idx, row in clusters_without_images.iterrows()]
# Extract midpoints of the edges with images
cluster_points_with_images = [row['representative_point'] for idx, row in clusters_with_images.iterrows()]

#fig, ax = ox.plot_graph(G, node_size=1, figsize=(60, 60), show=False, close=False) 

# Plotting each pair with unique color and drawing a line between them
#for i in range(len(actual_cluster_points)):
#    ax.plot([actual_cluster_points[i].x, closest_cluster_points[i].x], 
#            [actual_cluster_points[i].y, closest_cluster_points[i].y], 
#            c='lemonchiffon', linewidth=0.5)
    # highlight the actual cluster
#    ax.scatter(actual_cluster_points[i].x, actual_cluster_points[i].y, c='indianred', s=1.2)

# Plot the midpoints of the edges with images in blue
#for point in cluster_points_with_images:
#    ax.scatter(point.x, point.y, c='green', s=2)

#plt.show()


####################
## Sanity check 2/2:
## Distribution of distances between missing clusters and their closest clusters
####################

# Calculate distances
distances = []
for i, row in clusters_without_images.iterrows():
    dist = euclidean((row['representative_point'].x, row['representative_point'].y),
                     (row['closest_representative_point'].x, row['closest_representative_point'].y))
    distances.append(dist)


'''
A degree of latitude is approximately 111 kilometers.
Since the distances are small, we can use a simple approximation for converting degrees
to meters for both latitude and longitude, keeping in mind that this approximation becomes
less accurate for larger distances or distances calculated at high latitudes.
''' 

# Convert distances from degrees to meters
distances_meters = [dist * 111000 for dist in distances] 
    
# Plotting the distribution of distances
plt.figure(figsize=(10, 6))
sns.histplot(distances_meters, kde=True)
plt.title('Distribution of Distances Between Missing Clusters and Their Closest Clusters')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()

# Displaying statistical summary
pd.DataFrame(distances_meters, columns=['Distances']).describe()

''''
For Stuttgart, the distribution of distances between missing clusters and their closest clusters is right-skewed,
The mean distance is 186.4 meters, while the median is 116.2 meters and the standard deviation is 196.6 meters.
The maximum distance is 2.7 kilometers, which is extremely high compared to the rest of the distances.
Given the context of the project, the median seems to be a sensible threshold for deciding
whether to use the closest cluster's image.
'''


##################################################################
#### Assigning the closest cluster's ID based on distance treshold
##################################################################

# Use the median distance as a threshold for deciding whether to use the closest cluster's image
median_distance = np.median(distances)

# Implementation
centroids_nearest['implied_cluster_id'] = None  # Initialize with None

for i, row in centroids_nearest.iterrows():
    if row['cluster_id'] in retrieved_cluster_ids:
        centroids_nearest.at[i, 'implied_cluster_id'] = row['cluster_id']
    else:
        cluster_info = clusters_without_images[clusters_without_images['cluster_id'] == row['cluster_id']].iloc[0]
        dist = euclidean((row['representative_point'].x, row['representative_point'].y),
                         (cluster_info['closest_representative_point'].x, cluster_info['closest_representative_point'].y))
        if dist <= median_distance:
            centroids_nearest.at[i, 'implied_cluster_id'] = cluster_info['closest_cluster_id']
        else:
            centroids_nearest.at[i, 'implied_cluster_id'] = pd.NA  # Assign NA if distance exceeds threshold

# Check how many missing clusters are left
centroids_nearest['implied_cluster_id'].isna().sum()

# Write the updated dataframe to a file
centroids_nearest.to_pickle(f"processed/clusters_processed_{place_name.split(',')[0]}.pkl")
