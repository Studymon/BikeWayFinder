import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

import os
import zipfile
import gzip
import io

#####################################
#### Image Feature Pipeline Part 5/5:
#### Clustering + SVI Extraction
#### Sorting SVI [notebook]
#### Image Segmentation [notebook]
#### Missing Cluster Handling
#### >> Image Feature Extraction <<
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
os.chdir('../../data')

place_name = "Stuttgart, Germany"

# Import the clustered edges and cluster information [see src/features/svi_extraction.py]
edges_clustered = pd.read_pickle(f"interim/edges_{place_name.split(',')[0]}_clustered.pkl")
nodes = pd.read_pickle(f"interim/nodes_{place_name.split(',')[0]}.pkl")  # for building the graph G later
clusters_processed = pd.read_pickle(f"processed/clusters_processed_{place_name.split(',')[0]}.pkl")

# Drop unused columns
centroids_nearest = centroids_nearest.drop(columns=['DBSCAN_group'])


##############################
#### Extracting image features
##############################
