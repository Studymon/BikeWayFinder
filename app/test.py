import folium
import requests
import streamlit as st 
from streamlit_folium import folium_static
import osmnx as ox
import networkx as nx

# Getting start/dest coordinates
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
    return G

G = init_osm_graph('Stuttgart')
    
# Get fastest route from OSM
def get_osm_route(start_location, dest_location):
     
    start_data = get_lat_lon(start_location)
    dest_data = get_lat_lon(dest_location)

    orig_node = ox.distance.nearest_nodes(G, start_data[1], start_data[0])
    dest_node = ox.distance.nearest_nodes(G, dest_data[1], dest_data[0])
    shortest_route = nx.shortest_path(G, orig_node, dest_node, weight="length")
    
    return shortest_route

   
# App layout
APP_TITLE = 'BikeWayFinder'
APP_SUBTITLE = 'Find the best way by bike from A to B'
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Input for start/dest location
start_location = st.text_input('Enter start location:')
dest_location = st.text_input('Enter destination:')

# Dropdown for route type
route_type = st.selectbox('Select route type:', ['Shortest Route', 'Bike-Friendly Route'])

# Button to trigger route calculation
if st.button('Find Route'):
    start_data = get_lat_lon(start_location)
    dest_data = get_lat_lon(dest_location)

    # Create folium map
    m = folium.Map(location=start_data, zoom_start=11)

    # Add markers and polyline
    folium.Marker(start_data, popup='Start').add_to(m)
    folium.Marker(dest_data, popup='Destination').add_to(m)
    
    # Convert route type to lowercase for consistency
    route_type_lower = route_type.lower()

    if route_type_lower == 'bike-friendly route':
        # Just a straight polyline
        folium.PolyLine([start_data, dest_data]).add_to(m)
    else:      
        # Get the route based on the selected type
        shortest_route = get_osm_route(start_location, dest_location)
         
        m = ox.plot_route_folium(G, shortest_route, tiles='openstreetmap')
        folium.Marker(start_data, popup='Start',
                    icon = folium.Icon(color='green', prefix='fa',icon='bicycle')).add_to(m)
        folium.Marker(dest_data, popup='Destination', 
                    icon = folium.Icon(color='red', icon="flag")).add_to(m)

    # Display the map
    folium_static(m, width=700)
else:
    # Display an empty map
    folium_static(folium.Map(location=[48.7758, 9.1829], zoom_start=12), width=700)