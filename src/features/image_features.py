import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

import torch

import os
import re
import gzip
import io
import json

#####################################
#### Image Feature Pipeline Part 5/5:
#### Clustering + SVI Extraction
#### Sorting SVI [notebook]
#### Image Segmentation [notebook]
#### Missing Cluster Handling
#### >> Image Feature Extraction <<
######################################


##############
#### Functions
###############

# Function for importing the compressed tensors containing image segmentation information
def load_compressed_tensor(filename):
    with gzip.open(filename, 'rb') as f:
        buffer = f.read()
    tensor = torch.load(io.BytesIO(buffer))
    return tensor

# Function to compute the relative frequency of a class in a tensor
def compute_relative_frequency(tensor, class_id):
    total_pixels = tensor.numel()
    class_pixels = (tensor == class_id).sum().item()
    return class_pixels / total_pixels

# Function to compute the relative frequency of multiple classes in a tensor grouped together
def compute_relative_frequency_grouped(tensor, class_ids):
    total_pixels = tensor.numel()
    class_pixels = sum((tensor == class_id).sum().item() for class_id in class_ids)
    return class_pixels / total_pixels

# Function to detect the presence of a class in a tensor, given a threshold
def detect_class_presence(tensor, class_id, threshold=1):
    class_pixels = (tensor == class_id).sum().item()
    return 1 if class_pixels >= threshold else 0

# Function to normalize the relative frequency of a class, capping at a maximum value for outliers
def min_max(x, max_value, min_value=0):
    return (x - min_value) / (max_value - min_value) if x <= max_value else 1


################
#### Preparation
################

# Change to the data directory
os.chdir('../../data')

place_name = "Stuttgart, Germany"

# Import the clustered edges and cluster information [see src/features/svi_extraction.py]
edges_clustered = pd.read_pickle(f"interim/edges_{place_name.split(',')[0]}_clustered.pkl")
nodes = pd.read_pickle(f"interim/nodes_{place_name.split(',')[0]}.pkl")  # for building the graph G later
clusters_processed = pd.read_pickle(f"processed/clusters_processed_{place_name.split(',')[0]}.pkl")
survey_predictions = pd.read_csv(f"processed/new_edges_predictions.csv")

# Load pixel classes from Mapillary Vistas dataset
data = json.load(open('external/config_v1.2.json'))
'''
Note: This file is from the Mapillary Vistas dataset, which is licensed under CC BY 4.0.
The file is available at https://www.mapillary.com/dataset/vistas
The file contains the class labels and color palette that was used for the image segmentation.
'''

# Create dictionary of labels of pixel classes
labels = {}
for i, label in enumerate(data['labels']):
    labels[i] = label['readable']


##############################
#### Extracting image features
##############################

# Directory where the pred tensors are saved
pred_output_dir = "interim/edges_pred_output"
tensor_files = os.listdir(pred_output_dir)

#### TEST
# Load in example tensor
#tensor_path = os.path.join(pred_output_dir, 'edges-cluster_17335_pred.pt.gz')
#pred_tensor = load_compressed_tensor(tensor_path)

# Print how many pixels are there of classes of interest together with its readable label
#for class_id in [44, 12, 7, 10, 31, 30, 29, 55, 54, 61]:
#    print(f"Class {class_id} ({labels[class_id]}): {(pred_tensor == class_id).sum().item()}")
#### TEST END

'''
The classes we are interese in are:
- 12: Rail Track (Detection with threshold)
- 44: Street Light (Detection)

- 29: Terrain (Relatice Frequency)
- 30: Vegetation (Relatice Frequency)

- 54: Bus (Relatice Frequency)
- 55: Car (Relatice Frequency)
- 61: Truck (Relatice Frequency)

Both 29 and 30 and 54, 55 and 61 are grouped together as they are related to the same feature.
'''

# List to store the results for each tensor
results = []

for tensor_file in tensor_files:
    match = re.search(r"edges-cluster_(\d+)_pred.pt.gz", tensor_file)
    if match:
        cluster_number = int(match.group(1))
        
        # Adjust the path and loading method for compressed tensors
        tensor_path = os.path.join(pred_output_dir, tensor_file)
        pred_tensor = load_compressed_tensor(tensor_path)
        
        # Initialize a dictionary to hold the results for this tensor
        tensor_results = {
            'cluster_id': cluster_number,
        }
        
        # Compute results for the classes of interest
        tensor_results['rail_track_presence'] = detect_class_presence(pred_tensor, 12, threshold=2560)  # 1% of the image
        tensor_results['street_light_presence'] = detect_class_presence(pred_tensor, 44)
        tensor_results['greenery_rel_freq'] = compute_relative_frequency_grouped(pred_tensor, [29, 30])
        tensor_results['vehicles_rel_freq'] = compute_relative_frequency_grouped(pred_tensor, [54, 55, 61])

        results.append(tensor_results)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

## Inspect results
print(results_df.head())

# For relative frequencies, look at rows of the highest values
print(results_df.nlargest(15, 'vehicles_rel_freq'))
print(results_df.nlargest(15, 'greenery_rel_freq'))

# Look at maximum values the classes with relative frequency
print(results_df['greenery_rel_freq'].max())
print(results_df['vehicles_rel_freq'].max())

'''
Inspecting the results (including looking at the pictures with the highest values),
we can see that the relative frequency of greenery can be very high, in the context
of the images this makes sense.
The relative frequency of vehicles can also be relatively high, the maximum values
are outliers and results of unfortunate camera angles. It was determined that the
highest sensible relative frequency of vehicles is 0.35, which will serve as the
maximum value for min-max scaling, all higher values (12 of 12532 images) will be capped at 1.
'''
# Apply the min-max scaling
results_df['greenery_rel_freq'] = results_df['greenery_rel_freq'] / results_df['greenery_rel_freq'].max()
results_df['vehicles_rel_freq'] = results_df['vehicles_rel_freq'].apply(min_max, args=(0.35,))


##################################
#### Processing Survey Predictions
##################################

# Transform the filename to cluster_id
survey_predictions['cluster_id'] = survey_predictions['Filename'].apply(
    lambda x: int(re.search(r"edges-cluster_(\d+).jpg", x).group(1)))

# Keep only relevant columns
survey_predictions = survey_predictions[['cluster_id', 'Prediction']].copy()

# Min-max scaling for survey predictions
survey_min = survey_predictions['Prediction'].min()
survey_max = survey_predictions['Prediction'].max()
survey_predictions['Prediction'] = survey_predictions['Prediction'].apply(min_max, args=(survey_max, survey_min))


####################################
#### Merging Image Features to Edges
####################################

# First, merge the survey predictions to the results
survey_predictions.rename(columns={'Prediction': 'survey_score_prediction'}, inplace=True)
results_df = results_df.merge(survey_predictions, on='cluster_id', how='inner')

# Before merging, rename `cluster_id` of clusters_processed into `original_cluster_id`
clusters_processed.rename(columns={'cluster_id': 'original_cluster_id'}, inplace=True)

# Merge the results with the processed clusters
clusters_img_features = clusters_processed.merge(results_df,
                                                 left_on='implied_cluster_id',
                                                 right_on='cluster_id',
                                                 how='left')

# Keep only relevant columns
clusters_img_features.drop(columns=['centroid',
                                    'representative_point',
                                    'implied_cluster_id',
                                    'cluster_id'], inplace=True)

# Keep only relevant columns in edges_clustered
edges_clustered_keep = edges_clustered[['osmid', 'geometry', 'midpoint', 'cluster_id']]

# Preserve the original multi-index while merging by resetting it and keeping it as columns
edges_clustered_reset = edges_clustered_keep.reset_index()

# Merge image features to edges
edges_img_features = edges_clustered_reset.merge(clusters_img_features.rename(columns={'original_cluster_id': 'cluster_id'}),
                                                 how='left', on='cluster_id')
# Set index to u, v, key again post merge
edges_img_features.set_index(['u', 'v', 'key'], inplace=True)

# Save the results
edges_img_features.to_pickle(f"processed/edges_img_features_{place_name.split(',')[0]}.pkl")
