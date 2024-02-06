import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio

import osmnx as ox
import geopandas as gpd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from scipy.spatial import cKDTree

##############################
## READING ELEVATION RASTER
##############################

elv_raster = rio.open('elevation_raster.tif')
elv_data = elv_raster.read(1)

##############################
## LOADING OSM DATA
##############################

# Extract the graph from OSM
G = ox.graph_from_place('Stuttgart', network_type='bike')
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Extract coordinates from linestrings of edges
# Interpolate midpoint og linestrings
coords = edges['geometry'].apply(lambda geom: geom.interpolate(0.5, normalized=True) if isinstance(geom, LineString) else geom)
coords = pd.DataFrame(coords, columns=['geometry'])

# Extract coordinates
coords['longitude'], coords['latitude'] = zip(*[(point.xy[0][0], point.xy[1][0]) for point in coords.geometry])