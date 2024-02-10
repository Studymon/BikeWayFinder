import folium
import requests
import streamlit as st 
from streamlit_folium import folium_static
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely import wkt

# Getting start/dest coordinates
@st.cache_resource
def get_lat_lon(streetname):
    BASE_URL = 'https://nominatim.openstreetmap.org/search?format=json'
    response = requests.get(f'{BASE_URL}&street={streetname}&city=Stuttgart')
    data = response.json()
    
    if data:
        lat = data[0].get('lat')
        lon = data[0].get('lon')
        return float(lat), float(lon)
    else:
        # Handle the case where the geocoding service does not return valid data
        return None
 
# Initializing OSM Graph
@st.cache_resource
def init_osm_graph(city):
    G = ox.graph_from_place(city, network_type='bike')
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    return G, nodes

G, nodes = init_osm_graph('Stuttgart')

# Importing csv with calculated scores
@st.cache_data
def import_data(path):
    merged_osm = pd.read_csv(path)
    merged_osm = gpd.GeoDataFrame(merged_osm).set_index(["u", "v", "key"])
    merged_osm["geometry"] = merged_osm["geometry"].astype(str).apply(lambda x: wkt.loads(x))
    merged_osm = merged_osm.set_geometry('geometry')
    return merged_osm

merged_osm = import_data('scores_no_highways.csv')

# Initializing OSM_bike Graph
# @st.cache(allow_output_mutation=True, hash_funcs={gpd.GeoDataFrame: lambda _: None})
@st.cache_resource
def bike_graph(_edges):
    edges = gpd.GeoDataFrame(_edges).set_crs("epsg:4326")
    G = ox.graph_from_gdfs(nodes, _edges)
    return edges, G

bike_edges, G_bike = bike_graph(merged_osm)
    
# Get shortest route from OSM
@st.cache_resource
def get_osm_route(start_location, dest_location):
     
    start_data = get_lat_lon(start_location)
    dest_data = get_lat_lon(dest_location)

    orig_node = ox.distance.nearest_nodes(G, start_data[1], start_data[0])
    dest_node = ox.distance.nearest_nodes(G, dest_data[1], dest_data[0])
    shortest_route = nx.shortest_path(G, orig_node, dest_node, weight="length")
    pathDistance = nx.shortest_path_length(G, orig_node, dest_node, weight='length')
    
    return shortest_route, pathDistance

# Get best bike route from OSM
@st.cache_resource
def get_bike_route(_graph, start_location, dest_location, weight):
     
    start_data = get_lat_lon(start_location)
    dest_data = get_lat_lon(dest_location)

    orig_edge = ox.nearest_edges(_graph, start_data[1], start_data[0])
    dest_edge = ox.nearest_edges(_graph, dest_data[1], dest_data[0])
    best_route = ox.shortest_path(_graph, orig_edge[0], dest_edge[1], weight=weight)
    # Calculate the total length of the path using the 'length' attribute of the edges
    bike_pathDistance = sum(_graph[u][v][0]['length'] for u, v in zip(best_route[:-1], best_route[1:]))
    
    return best_route, bike_pathDistance

# Calculate the midpoint between start & destination
@st.cache_resource
def calculate_midpoint(start_data, dest_data):
    mid_lat = (start_data[0] + dest_data[0]) / 2
    mid_lon = (start_data[1] + dest_data[1]) / 2
    return (mid_lat, mid_lon)

# Function to calculate bikeability score
@st.cache_resource
def calculate_bikeability_score(_graph, route):
    scores = []
    # Iterate over the route to get pairs of nodes (start, end) representing each edge
    for start, end in zip(route[:-1], route[1:]):
        try:
            edge_data = _graph[start][end][0]
            score = edge_data.get('weightedFinalScore', None)
            if score is not None:
                scores.append(score)
        except KeyError:
            # Means we have a motorway, primary road, etc. in the route
            score = 0.2
            scores.append(score)
    mean_score = sum(scores) / len(scores)
    return mean_score


   
# App layout
APP_TITLE = 'BikeWayFinder'
APP_SUBTITLE = 'Find the best way by bike from A to B'
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Input for start/dest location
start_location = st.text_input('Enter start location:')
dest_location = st.text_input('Enter destination:')

# Dropdown for route type
route_type = st.selectbox('Select route type:', ['Shortest Route', 
                                                 'Bike-Friendly Route',
                                                 'Compare Routes'])

# Button to trigger route calculation
if st.button('Find Route'):
    start_data = get_lat_lon(start_location)
    dest_data = get_lat_lon(dest_location)
    
    # Calculate the midpoint
    midpoint = calculate_midpoint(start_data, dest_data)

    # Create folium map
    m = folium.Map(location=midpoint, zoom_start=13)
    
    # Get the best routes
    bikeable_route, bike_pathDistance = get_bike_route(G_bike, start_location, dest_location, "weightedFinalScore_reversed")
    bike_geom = [(G_bike.nodes[node]['y'], G_bike.nodes[node]['x']) for node in bikeable_route]
    shortest_route, pathDistance = get_osm_route(start_location, dest_location)
    route_geom = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in shortest_route]
    
    # Calculate bikeability score
    bike_score = calculate_bikeability_score(G_bike, bikeable_route)
    short_score = calculate_bikeability_score(G_bike, shortest_route)

    # Convert route type to lowercase for consistency
    route_type_lower = route_type.lower()

    if route_type_lower == 'compare routes':
        folium.PolyLine(bike_geom, color="blue", weight=4, opacity=1).add_to(m)
        folium.Marker(start_data, popup='Start',
                      icon = folium.Icon(color='green', prefix='fa',icon='bicycle')).add_to(m)
        folium.Marker(dest_data, popup='Destination', icon = folium.Icon(color='red', icon="flag")).add_to(m)

        # Fetch and display the shortest route
        folium.PolyLine(route_geom, color="red", weight=4, opacity=0.5).add_to(m)
    
    elif route_type_lower == 'bike-friendly route':
        folium.PolyLine(bike_geom, color="blue", weight=4, opacity=1).add_to(m)
        folium.Marker(start_data, popup='Start',
                      icon = folium.Icon(color='green', prefix='fa',icon='bicycle')).add_to(m)
        folium.Marker(dest_data, popup='Destination', icon = folium.Icon(color='red', icon="flag")).add_to(m)
    else:      
        # Get the shortest route
        folium.PolyLine(route_geom, color="blue", weight=4, opacity=1).add_to(m)
        folium.Marker(start_data, popup='Start',
                      icon = folium.Icon(color='green', prefix='fa',icon='bicycle')).add_to(m)
        folium.Marker(dest_data, popup='Destination', icon = folium.Icon(color='red', icon="flag")).add_to(m)

    
    # Display route information prominently using Streamlit widgets
    if route_type_lower == 'bike-friendly route':
        st.markdown(f"#### Bike-Friendly Route Details")
        st.write(f"**Distance:** {round(bike_pathDistance/1000, 2)} km")
        st.write(f"**Estimated Time Needed:** {round((bike_pathDistance/1000 / 15) * 60)} minutes")
    elif route_type_lower == 'shortest route':
        st.markdown(f"#### Shortest Route Details")
        st.write(f"**Distance:** {round(pathDistance/1000, 2)} km")
        st.write(f"**Estimated Time Needed:** {round((pathDistance/1000 / 15) * 60)} minutes")
    elif route_type_lower == 'compare routes':
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### Bike-Friendly Route")
            st.write(f"**Distance:** {round(bike_pathDistance/1000, 2)} km")
            st.write(f"**Estimated Time Needed:** {round((bike_pathDistance/1000 / 15) * 60)} minutes")
            st.write(f"**Average Bikeability Score:** {round(bike_score*100, 2)} %")
        with col2:
            st.markdown(f"#### Shortest Route")
            st.write(f"**Distance:** {round(pathDistance/1000, 2)} km")
            st.write(f"**Estimated Time Needed:** {round((pathDistance/1000 / 15) * 60)} minutes")
            st.write(f"**Average Bikeability Score:** {round(short_score*100, 2)} %")
        
        st.markdown("""
        <style>
        .legend {
            display: flex;
            align-items: center;
        }
        .color-box {
            width: 12px;
            height: 12px;
            margin-right: 5px;
            border: 1px solid #888;
        }
        .blue-box {
            background-color: blue;
        }
        .red-box {
            background-color: red;
        }
        </style>
        <div class="legend">
            <div class="color-box blue-box"></div><span>- Bike-friendly Route</span>
        </div>
        <div class="legend">
            <div class="color-box red-box"></div><span>- Shortest Route</span>
        </div>
        """, unsafe_allow_html=True)
        
    
    # Display the map
    folium_static(m, width=700)
    
else:
    # Display an empty map
    folium_static(folium.Map(location=[48.7758, 9.1829], zoom_start=12), width=700)
