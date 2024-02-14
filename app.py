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

merged_osm = import_data('data_visible/processed/final_scores.csv')

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
    if start_data is None or dest_data is None:
        return None, "Start or destination location could not be resolved."
    try:
        orig_edge = ox.nearest_edges(_graph, start_data[1], start_data[0])
        dest_edge = ox.nearest_edges(_graph, dest_data[1], dest_data[0])
        best_route = ox.shortest_path(_graph, orig_edge[0], dest_edge[1], weight=weight)
    except Exception as e:
        # If bike-friendly route is not found, fallback to shortest route
        print(f"Failed to find bike-friendly path: {e}. Falling back to shortest route.")
        orig_node = ox.distance.nearest_nodes(G, start_data[1], start_data[0])
        dest_node = ox.distance.nearest_nodes(G, dest_data[1], dest_data[0])
        best_route = nx.shortest_path(G, orig_node, dest_node, weight="length")
    if best_route is None:
        return None, "No path found."
    bike_pathDistance = sum(_graph[u][v][0]['length'] for u, v in zip(best_route[:-1], best_route[1:]))
    
    return best_route, bike_pathDistance


# Calculate the midpoint between start & destination
@st.cache_resource
def calculate_midpoint(start_data, dest_data):
    # Check if start_data or dest_data is None and return an appropriate message
    if start_data is None and dest_data is None:
        return "Both start and destination locations do not exist."
    elif start_data is None:
        return "The start location does not exist."
    elif dest_data is None:
        return "The destination location does not exist."

    # If both start_data and dest_data are valid, calculate the midpoint
    mid_lat = (start_data[0] + dest_data[0]) / 2
    mid_lon = (start_data[1] + dest_data[1]) / 2
    return (mid_lat, mid_lon)

# Function to calculate bikeability score
@st.cache_resource
def calculate_bikeability_score(_graph, route, weight_param):
    if route is None:
        return "Unable to calculate score without a valid route."
    scores = []
    for start, end in zip(route[:-1], route[1:]):
        try:
            edge_data = _graph[start][end][0]
            score = 1 - edge_data.get(weight_param, None)
            if score is not None:
                scores.append(score)
        except KeyError:
            score = 0  # Assuming a default score for non-bike-friendly paths
            scores.append(score)
    mean_score = sum(scores) / len(scores) if scores else 0  # Avoid division by zero
    return mean_score


# Function to display bikeability score as bars
@st.cache_resource
def render_score_as_bars(score):

    full_bars = int(score * 10)
    half_bar_needed = (score * 10) % 1 >= 0.5
    percentage_score = round(score * 100, 2)
    
    percentage_html = f"<div style='margin-bottom: 5px;'><strong>Average Bikeability Score:</strong> {percentage_score}%</div>"
    
    bars_html = ""
    for i in range(1, 11):
        if i <= full_bars:
            color = "green"
            bars_html += f'<div style="width: 20px; height: 20px; display: inline-block; background-color: {color}; margin-right: 2px;"></div>'
        elif i == full_bars + 1 and half_bar_needed:
            # Use linear-gradient to create the half-bar effect
            bars_html += f'<div style="width: 20px; height: 20px; display: inline-block; background: linear-gradient(to right, green 50%, grey 50%); margin-right: 2px;"></div>'
        else:
            color = "grey"
            bars_html += f'<div style="width: 20px; height: 20px; display: inline-block; background-color: {color}; margin-right: 2px;"></div>'
    
    # Combine the percentage and bars HTML
    combined_html = percentage_html + bars_html
    
    # Display the combined HTML content
    st.markdown(combined_html, unsafe_allow_html=True)



   
# App layout
APP_TITLE = 'BikeWayFinder'
APP_SUBTITLE = 'Find the best way by bike from A to B'
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Input for start/dest location
start_location = st.text_input('Enter start location:')
dest_location = st.text_input('Enter destination:')

# Dropdown for route type
route_type = st.selectbox('Select route type:', ['Bike-Friendly Route',
                                                 'Shortest Route',
                                                 'Compare Routes'])

# Radio button for selecting weight parameter
weight_options = {
    'I trust my App developers (default)': 'baselineScore_reversed',
    'I do not like traffic and prefer riding in nature': 'natureScore_reversed',
    'I highly trust the opinions of my fellow cyclists': 'perceptionScore_reversed'
}
selected_weight = st.radio('Customize Your Bike Journey:', list(weight_options.keys()))

# Button to trigger route calculation
if st.button('Find Route'):
    start_data = get_lat_lon(start_location)
    dest_data = get_lat_lon(dest_location)
    
    # Use the selected weight parameter from the radio button
    weight_param = weight_options[selected_weight]
    
    execution_continue = True
    
    # Calculate the midpoint
    midpoint = calculate_midpoint(start_data, dest_data)
    
    # Check if the midpoint is a tuple (coordinates) or a string (error message)
    if not isinstance(midpoint, tuple):
        st.error(midpoint)
        execution_continue = False
    else:
        m = folium.Map(location=midpoint, zoom_start=13)

    if execution_continue:
        
        # Attempt to get the bike-friendly route
        bikeable_route, bike_pathDistance = get_bike_route(G_bike, start_location, dest_location, weight_param)
        
        # Always fetch the shortest route for comparison or fallback
        shortest_route, pathDistance = get_osm_route(start_location, dest_location)
        route_geom = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in shortest_route]
        short_score = calculate_bikeability_score(G_bike, shortest_route, weight_param)
        
        # Determine if we need to fallback to the shortest route
        fallback_to_shortest = bikeable_route is None
        
        if not fallback_to_shortest:
            # If we have a bikeable route, use it
            bike_geom = [(G_bike.nodes[node]['y'], G_bike.nodes[node]['x']) for node in bikeable_route]
            bike_score = calculate_bikeability_score(G_bike, bikeable_route, weight_param)
        else:
            # If no bikeable route, fallback to shortest and consider its score
            bike_geom = route_geom  
            bike_pathDistance = pathDistance 
            bike_score = short_score
            st.info("Failed to find a bike-friendly route up to our standards. Displaying the shortest route instead.")
            
            
        # Calculate estimated time in minutes for each route (avg bike speed 15km/h)
        bike_time_estimated = (bike_pathDistance / 1000 / 15) * 60
        shortest_time_estimated = (pathDistance / 1000 / 15) * 60

        # Determine if the bike route is preferable
        prefer_bike_route = True 
        
        # Check if bikeability score for the bike route is lower than the shortest route
        if bike_score < short_score:
            prefer_bike_route = False
        # Check if bikeability score for the bike route is within 2 percentage points of the shortest route
        # and the time needed is longer by more than 5 minutes
        elif abs(bike_score - short_score) <= 0.02 and (bike_time_estimated - shortest_time_estimated) > 5:
            prefer_bike_route = False
            
        # Based on the decision, adjust the route to be used for display and further calculations
        if prefer_bike_route:
            # Use bikeable_route for further actions
            bikeable_route = bikeable_route
            bike_geom = bike_geom
            bike_pathDistance = bike_pathDistance
            bike_time_estimated = bike_time_estimated
            bike_score = bike_score
        else:
            # Fallback to shortest_route
            bikeable_route = shortest_route
            bike_geom = route_geom
            bike_pathDistance = pathDistance
            bike_time_estimated = shortest_time_estimated
            bike_score = short_score
            st.info("The shortest route is the most bicycle-friendly")
                

        
        # Convert route type to lowercase for consistency
        route_type_lower = route_type.lower()

        if route_type_lower == 'compare routes' or fallback_to_shortest or prefer_bike_route:
            folium.PolyLine(bike_geom, color="blue", weight=4, opacity=1).add_to(m)
            folium.Marker(start_data, popup='Start',
                        icon = folium.Icon(color='green', prefix='fa',icon='bicycle')).add_to(m)
            folium.Marker(dest_data, popup='Destination', icon = folium.Icon(color='red', icon="flag")).add_to(m)

            # Fetch and display the shortest route
            folium.PolyLine(route_geom, color="red", weight=4, opacity=0.5).add_to(m)
        
        elif route_type_lower == 'bike-friendly route' or fallback_to_shortest or prefer_bike_route:
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

        # Display the route details
        if route_type_lower == 'bike-friendly route' or fallback_to_shortest:
            st.markdown(f"#### Bike-Friendly Route Details")
            st.write(f"**Distance:** {round(bike_pathDistance / 1000, 2)} km")
            st.write(f"**Estimated Time Needed:** {round(bike_time_estimated)} minutes")
        elif route_type_lower == 'shortest route':
            st.markdown(f"#### Shortest Route Details")
            st.write(f"**Distance:** {round(pathDistance / 1000, 2)} km")
            st.write(f"**Estimated Time Needed:** {round(shortest_time_estimated)} minutes")
        elif route_type_lower == 'compare routes' or fallback_to_shortest:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### Bike-Friendly Route")
                st.write(f"**Distance:** {round(bike_pathDistance / 1000, 2)} km")
                st.write(f"**Estimated Time Needed:** {round(bike_time_estimated)} minutes")
                render_score_as_bars(bike_score)
            with col2:
                st.markdown(f"#### Shortest Route")
                st.write(f"**Distance:** {round(pathDistance / 1000, 2)} km")
                st.write(f"**Estimated Time Needed:** {round(shortest_time_estimated)} minutes")
                render_score_as_bars(short_score)
                
            # Skip before the legend
            st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
            
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
