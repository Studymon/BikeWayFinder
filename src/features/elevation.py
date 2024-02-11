import numpy as np
import pandas as pd
import rasterio as rio
import os
from pathlib import Path
import osmnx as ox
from shapely.geometry import LineString
from scipy.interpolate import griddata

#################################
## SET PATHS
#################################

BASE_DIR = Path(os.getcwd()).parent.parent
RASTER_DIR = os.path.join(BASE_DIR, 'src', 'data', 'elevation_raster.tif')

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
coords['elevation'] = 0



##############################
## MAPPING ELEVATION TO OSM
##############################    
    
with rio.open(RASTER_DIR) as dataset:
    for index, row in coords.iterrows():
        for val in dataset.sample([(row['longitude'], row['latitude'])]):
            coords.at[index, 'elevation'] = val[0]
            
# -9999 as NaN
coords['elevation'] = coords['elevation'].apply(lambda x: np.nan if x == -9999 else x)



##############################
## INTERPOLATING MISSING ELEVATION
##############################

# Perform interpolation using griddata from scipy
coords['elevation_interpolated'] = griddata(coords.dropna(subset=['elevation'])[['longitude', 'latitude']],
                                            coords.dropna(subset=['elevation'])[['elevation']],
                                            coords[['longitude', 'latitude']],
                                            method='linear')

# Check if there are still na values after linear interpolation
if coords['elevation_interpolated'].isna().sum() > 0:
    # Use 'nearest' method for points that could not be interpolated with 'linear'
    coords['elevation_interpolated'] = coords.apply(
        lambda row: row['elevation_interpolated'] if not pd.isna(row['elevation_interpolated']) 
        else griddata(
            coords.dropna(subset=['elevation'])[['longitude', 'latitude']],
            coords.dropna(subset=['elevation'])['elevation'],
            coords[['longitude', 'latitude']],
            method='nearest'
        )[0],
        axis=1
    )

# Replace the NaN values in the original elevation column with the interpolated values
coords['elevation'].fillna(coords['elevation_interpolated'], inplace=True)
coords.drop(columns=['elevation_interpolated'], inplace=True)



##############################
## CALCULATING ELEVATION GAIN PER EDGE
##############################

# Group by edges    
coords['elevation_gain'] = coords.groupby('u')['elevation'].transform(lambda x: x.iloc[-1] - x.iloc[0])

# Keeping only elevation variables
coords = coords[['elevation', 'elevation_gain']]



##############################
## EXPORTING DATA
##############################

OUT_DIR = os.path.join(BASE_DIR, 'src', 'data', 'elevation.csv')
coords.to_csv(OUT_DIR, index=True)