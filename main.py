from fastapi import FastAPI, Form, HTTPException, Body, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import folium
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
import networkx as nx
from shapely.geometry import LineString
import pickle
import mysql.connector
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from folium.plugins import HeatMap
from sklearn.neighbors import KernelDensity
import numpy as np
from ipywidgets import interact, widgets
from IPython.display import display
from joblib import dump, load
from typing import List
import psycopg2
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define a function to find the safest path between two nodes
def safest_path(graph, start_node, end_node):
    # Create a new graph with the same nodes and edges as the original graph
    new_graph = nx.Graph()
    new_graph.add_nodes_from(graph.nodes(data=True))
    new_graph.add_edges_from(graph.edges(data=True))

    # Update the edge weights in the new graph to use the crime_weight attribute
    # All edges without crime weight assigned to them are set default as 0 crime weight. 
    for u, v, data in new_graph.edges(data=True):
        data['weight'] = data.get('crime_weight', 0)

    # Use Dijkstra's algorithm to find the safest path between the start and end nodes
    safest_path = nx.dijkstra_path(new_graph, start_node, end_node, weight='weight')
    
    # Calculate the total weight of the edges along the safest path
    total_weight = 0
    for i in range(len(safest_path) - 1):
        u = safest_path[i]
        v = safest_path[i + 1]
        total_weight += new_graph[u][v]['weight']
    
    return safest_path, total_weight

# Define a function to find the safest path between two nodes
def shortest_path(graph, start_node, end_node):
    # Use Dijkstra's algorithm to find the shortest path between the start and end nodes
    shortest_path = nx.dijkstra_path(graph, start_node, end_node)
    
    # Create a new graph with the same nodes and edges as the original graph
    new_graph = nx.Graph()
    new_graph.add_nodes_from(graph.nodes(data=True))
    new_graph.add_edges_from(graph.edges(data=True))

    # Update the edge weights in the new graph to use the crime_weight attribute
    # All edges without crime weight assigned to them are set default as 0 crime weight. 
    for u, v, data in new_graph.edges(data=True):
        data['weight'] = data.get('crime_weight', 0)
    
    # Calculate the total weight of the edges along the shortest path based on the crime_weight attribute
    total_weight = 0
    for i in range(len(shortest_path) - 1):
        u = shortest_path[i]
        v = shortest_path[i + 1]
        total_weight += new_graph[u][v]['weight']
    
    return shortest_path, total_weight

# Define a function to visualize the safest path on a folium map
def visualize_safest_path(graph, start_node, end_node, map_obj):
    # Find the safest path between the start and end nodes
    path, _ = safest_path(graph, start_node, end_node)

    # Create a list of LineString objects representing the edges in the path
    path_edges = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_data = graph.get_edge_data(u, v)
        # Check if edge_data is None before trying to access its elements
        if edge_data is not None:
            if 'geometry' in edge_data[0]:
                edge_geom = edge_data[0]['geometry']
                path_edges.append(edge_geom)
            else:
                x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
                x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
                edge_geom = LineString([(x1, y1), (x2, y2)])
                path_edges.append(edge_geom)
        else:
            # Handle the case where edge_data is None
            # For example, you could assign a default value or raise an exception
            # pass

            # Handle the case where edge_data is None
            x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
            x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
            edge_geom = LineString([(x1, y1), (x2, y2)])
            path_edges.append(edge_geom)


    # Add the edges to the folium map
    for edge in path_edges:
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in edge.coords],
            color='green',
            weight=5,
            opacity=1
        ).add_to(map_obj)

# Define a function to visualize the safest path on a folium map
def visualize_shortest_path(graph, start_node, end_node, map_obj):
    # Find the shortest path between the start and end nodes
    path, _  = shortest_path(graph, start_node, end_node)

    # Create a list of LineString objects representing the edges in the path
    path_edges = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_data = graph.get_edge_data(u, v)
        # Check if edge_data is None before trying to access its elements
        if edge_data is not None:
            if 'geometry' in edge_data[0]:
                edge_geom = edge_data[0]['geometry']
                path_edges.append(edge_geom)
            else:
                x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
                x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
                edge_geom = LineString([(x1, y1), (x2, y2)])
                path_edges.append(edge_geom)
        else:
            # Handle the case where edge_data is None
            x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
            x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
            edge_geom = LineString([(x1, y1), (x2, y2)])
            path_edges.append(edge_geom)

    # Add the edges to the folium map
    for edge in path_edges:
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in edge.coords],
            color='blue',
            weight=5,
            opacity=1
        ).add_to(map_obj)

def generate_safest_path_map(graph, start_node, end_node, start_lat, start_lon, end_lat, end_lon):
    m = folium.Map(location=[start_lat, start_lon], zoom_start=14, zoom_control=False)

    # Create draggable markers for start and end locations on the map
    folium.Marker([start_lat, start_lon], icon=folium.Icon(color='blue'), draggable=False).add_to(m)
    folium.Marker([end_lat, end_lon], icon=folium.Icon(color='red'), draggable=False).add_to(m)

    # Visualize the safest path on the map
    visualize_safest_path(graph, start_node, end_node, m)

    return m

def generate_shortest_path_map(graph, start_node, end_node, start_lat, start_lon, end_lat, end_lon):
    m = folium.Map(location=[start_lat, start_lon], zoom_start=14, zoom_control=False)

    # Create draggable markers for start and end locations on the map
    folium.Marker([start_lat, start_lon], icon=folium.Icon(color='blue'), draggable=False).add_to(m)
    folium.Marker([end_lat, end_lon], icon=folium.Icon(color='red'), draggable=False).add_to(m)

    # Visualize the shortest path on the map
    visualize_shortest_path(graph, start_node, end_node, m)

    return m

class PathRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

@app.post('/find_path', response_class=JSONResponse)
async def find_path(request_data: PathRequest):
    start_lat = request_data.start_lat
    start_lon = request_data.start_lon
    end_lat = request_data.end_lat
    end_lon = request_data.end_lon
    
    # Use pandas to execute SQL query and store result in DataFrame
    conn = create_conn()
    query = "SELECT * FROM crime_data;"  # replace with your actual SQL query
    crime_data = pd.read_sql_query(query, conn)

    # Get coordinates
    crime_coords = crime_data[['latitude', 'longitude']].values

    # Create KDTree
    kdtree = cKDTree(crime_coords)

    place_name = "Iloilo City, Western Visayas, 5000, Philippines"
    graph_file = 'graph.pickle'

    try:
        # Try to load the graph from a file using pickle.load
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)

    except FileNotFoundError:
        # If the file does not exist, create a new graph and assign weights to its edges
        graph = ox.graph_from_place(place_name, network_type="all_private")

        # Assign crime data to the nearest edges
        for _, row in crime_data.iterrows():
            crime_coords = (row['LATITUDE'], row['LONGITUDE'])
            _, idx = kdtree.query(crime_coords)  # Find the index of the nearest crime coordinate

            nearest_node = ox.nearest_nodes(graph, row['LONGITUDE'], row['LATITUDE'])  # Find the nearest graph node
            nearest_edge = ox.distance.nearest_edges(graph, row['LONGITUDE'], row['LATITUDE'])  # Find the nearest edge

            # Assign weights to the edges
            if 'crime_weight' not in graph[nearest_edge[0]][nearest_edge[1]][0]:
                graph[nearest_edge[0]][nearest_edge[1]][0]['crime_weight'] = row['WEIGHTS']
            else:
                graph[nearest_edge[0]][nearest_edge[1]][0]['crime_weight'] += row['WEIGHTS']

        # Save the graph with assigned weights to a file using pickle.dump for faster loading next time
        with open(graph_file, 'wb') as f:
            pickle.dump(graph, f)

    # Find the node IDs for these locations using the ox.nearest_nodes function
    start_node = ox.nearest_nodes(graph, start_lon, start_lat)
    end_node = ox.nearest_nodes(graph, end_lon, end_lat)

    # Generate the map with the safest path
    safest_path_map = generate_safest_path_map(graph, start_node, end_node, start_lat, start_lon, end_lat, end_lon)
    # Returns the total weight
    _, safest_path_weight = safest_path(graph, start_node, end_node)

    # Generate the map with the shortest path
    shortest_path_map = generate_shortest_path_map(graph, start_node, end_node, start_lat, start_lon, end_lat, end_lon)
    # Returns the total weight
    _, shortest_path_weight =shortest_path(graph, start_node, end_node)

    # Convert the maps to HTML and send them as JSON responses
    safest_path_map_html = safest_path_map._repr_html_().replace('<div', '<div style="height: 100vh;"')
    shortest_path_map_html = shortest_path_map._repr_html_().replace('<div', '<div style="height: 100vh;"')

    # Pass the nodes of the path to Google Maps
    path, _ = safest_path(graph, start_node, end_node)
    path_coords = []
    for node in path:
        node_data = graph.nodes[node]
        lat = node_data['y']
        lon = node_data['x']
        path_coords.append((lat, lon))

    # Construct a Google Maps URL that displays the safest path with waypoints node by node
    start = str(start_lat) + ',' + str(start_lon)
    end = str(end_lat) + ',' + str(end_lon)
    waypoints = '|'.join([str(lat) + ',' + str(lon) for lat, lon in path_coords])
    google_maps_url_safest = 'https://www.google.com/maps/dir/?api=1&origin=' + start + '&destination=' + end + '&travelmode=driving&waypoints=' + waypoints

    # Pass the nodes of the path to Google Maps
    path_shortest, _ = shortest_path(graph, start_node, end_node)
    path_coords_shortest = []
    for node in path_shortest:
        node_data = graph.nodes[node]
        lat = node_data['y']
        lon = node_data['x']
        path_coords_shortest.append((lat, lon))

    # Construct a Google Maps URL that displays the shortest path with waypoints node by node
    start = str(start_lat) + ',' + str(start_lon)
    end = str(end_lat) + ',' + str(end_lon)
    waypoints_shortest = '|'.join([str(lat) + ',' + str(lon) for lat, lon in path_coords_shortest])
    google_maps_url_shortest = 'https://www.google.com/maps/dir/?api=1&origin=' + start + '&destination=' + end + '&travelmode=driving&waypoints=' + waypoints_shortest

    print(safest_path_weight, shortest_path_weight)
    # Return JSON response with necessary data
    return {
        "safest_path_map_html": safest_path_map_html,
        "shortest_path_map_html": shortest_path_map_html,
        "google_maps_url_safest": google_maps_url_safest,
        "google_maps_url_shortest": google_maps_url_shortest,
        "safest_path_weight": safest_path_weight,
        "shortest_path_weight": shortest_path_weight
    }

# Add the CORS middleware configuration for the /find_path route
@app.middleware("http")
async def add_cors_header(request, call_next):
    # Allow requests from 'https://safetyoin2.onrender.com'
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response


class HeatmapInput(BaseModel):
    day: List[str] = []
    month: List[str] = []
    year: List[str] = []
    district: List[str] = []
    category: List[str] = []

@app.post("/heatmap")
async def create_heatmap(data: HeatmapInput):
    # Read the dataset and assign it to the variable 'df'
    df = pd.read_csv("2023_final_dataset.csv")

    # We need to convert the int YEAR to a string
    column_name = "YEAR"
    df[column_name] = df[column_name].astype(str)

    # Create a copy of the dataframe
    filtered_data = df.copy()

    # Apply filters based on user input
    if data.day:
        filtered_data = filtered_data[filtered_data['DAY'].isin(data.day)]
    if data.month:
        filtered_data = filtered_data[filtered_data['MONTH'].isin(data.month)]
    if data.year:
        filtered_data = filtered_data[filtered_data['YEAR'].isin(data.year)]
    if data.district:
        filtered_data = filtered_data[filtered_data['DISTRICT'].isin(data.district)]
    if data.category:
        filtered_data = filtered_data[filtered_data['CATEGORY'].isin(data.category)]

    num_rows = filtered_data.shape[0]

    # Display the number of rows
    print("Number of rows in the dataset:", num_rows)

    # Add the following line to specify bandwidth
    bandwidth = 0.01

    # Create a KDE model
    lat_lon = np.array(filtered_data[['LATITUDE', 'LONGITUDE']])

    # Create a Folium map centered on Iloilo City
    # Coordinates of the center of Iloilo City
    map_center = [10.720321, 122.562019]

    # Create a Folium map centered on Iloilo City
    iloilo_kde = folium.Map(location=map_center, zoom_start=12, zoom_control=False)

    # Iterate over different crime types with weights
    for weight in range(1, 17):
        crime_data = filtered_data[filtered_data['WEIGHTS'] == weight][['LATITUDE', 'LONGITUDE']]

        # Check if crime_data is not empty before fitting KernelDensity
        if not crime_data.empty:
            crime_kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(crime_data)

            # Evaluate the KDE for the crime type heatmap
            crime_positions = np.vstack([crime_data['LATITUDE'], crime_data['LONGITUDE']]).T
            crime_z = np.exp(crime_kde.score_samples(crime_positions))

            # Define the gradient for the heatmap
            gradient = {
                0.2: 'blue',
                0.4: 'green',
                0.6: 'yellow',
                0.8: 'orange',
                1.0: 'red'
            }

            # Add the crime heatmap to the map
            HeatMap(list(zip(crime_data['LATITUDE'], crime_data['LONGITUDE'], crime_z)),
                    min_opacity=0.2,
                    max_val=np.max(crime_z),
                    radius=15,
                    blur=10,
                    max_zoom=1,
                    overlay=True,
                    control=False,
                    gradient=gradient).add_to(iloilo_kde)
    
    iloilo_kde.save('crime_heatmap.html')
    
    # Return the map
    return {"html": iloilo_kde._repr_html_().replace('<div', '<div style="height: 100vh;"')}

#DATABASE
def create_conn():
    conn = psycopg2.connect(
        host='ep-noisy-glitter-a1knt3hq.ap-southeast-1.aws.neon.tech',  
        port=5432, 
        user='safetypin_owner',  
        password='cti63KwTmHOd',  
        dbname='safetypin' 
    )
    return conn

@app.get("/data")
def get_data(page: int = 0, page_size: int = 100):
    conn = create_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM crime_data OFFSET %s LIMIT %s", (page * page_size, page_size))
    rows = cur.fetchall()
    return {"data": rows}

@app.get("/alphabetical")
def get_alphabetical(page: int = 0, page_size: int = 100):
    conn = create_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM crime_data ORDER BY barangay ASC OFFSET %s LIMIT %s", (page * page_size, page_size))
    rows = cur.fetchall()
    return {"alphabetical": rows}


from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt
import psycopg2

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post('/login')
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password

    conn = create_conn()
    cur = conn.cursor()

    # Use a parameterized query to prevent SQL injection
    cur.execute("SELECT * FROM admin WHERE username=%s", (username,))
    user = cur.fetchone()

    conn.close()

    if not user or not password == user[4]: # 4 is the position of the column password in the table
        return {"detail": "Invalid username or password"}

    print(user)
    # Create a JWT token
    token_data = {"officer_id": user[0], "officer_name": user[1],  "officer_position": user[2],  "username": user[3]}  # numbers is index position of the data in the data base column
    token = jwt.encode(token_data, "secret-key", algorithm="HS256")

    return {"access_token": token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, "secret-key", algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

class User(BaseModel):
    officer_id: int
    officer_name: str
    officer_position: str
    username: str
    password: str

@app.post("/register")
async def register(user: User):
    print(user.officer_id)
    conn = create_conn()
    cur = conn.cursor()

    # Check if a user with the given officername already exists
    cur.execute("SELECT * FROM admin WHERE officer_name=%s", (user.officer_name,))
    existing_user = cur.fetchone()

    if existing_user is not None:
        # If a user with the given username already exists, return an error message
        return {"result": "User Exists."}

    else:
        # If no existing user is found, insert the new user into the database
        cur.execute("INSERT INTO admin (officer_id, officer_name, officer_position, username, password) VALUES (%s, %s, %s, %s, %s)", (user.officer_id, user.officer_name, user.officer_position, user.username, user.password))
        conn.commit()

        cur.close()
        conn.close()

        return {"result": "User Registered."}

@app.get("/users/me")
async def read_users_me(username: str = Depends(get_current_user)):
    print(username)
    return {"username": username}

@app.get("/")
async def read_root(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return {"message": "You are authorized"}

class CrimeData(BaseModel):
    district: str
    barangay: str
    dateReported: str
    timeReported: str
    dateCommitted: str
    timeCommitted: str
    offense: str
    category: str
    latitude: str
    longitude: str
    weight: str
    year: str
    month: str
    time: str
    lightCondition: str
    day: str
    officer: str

@app.post("/add")
async def add(data: CrimeData):
    print(data.officer)
    conn = create_conn()
    cur = conn.cursor()

    cur.execute("INSERT INTO crime_data (district, barangay, date_reported, time_reported, date_committed, time_committed, offenses, category, latitude, longitude, weights, year, month, time, light_condition, day, date_added, officer) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)", 
    (data.district, data.barangay, data.dateReported, data.timeReported, data.dateCommitted, data.timeCommitted, data.offense, data.category, data.latitude, data.longitude, data.weight, data.year, data.month, data.time, data.lightCondition, data.day, data.officer))
    conn.commit()

    cur.close()
    conn.close()

    return {"result": "Crime Data Added."}


class CrimeDataDelete(BaseModel):
    crimeID: int

@app.post("/delete")
async def delete(data: CrimeDataDelete):
    print(data.crimeID)
    conn = create_conn()
    cur = conn.cursor()

    # Check if the crime data exists
    cur.execute(f"SELECT * FROM crime_data WHERE crime_id={data.crimeID}")
    result = cur.fetchone()
    if result is None:
        return {"result": f"Crime data with the specified ID does not exist."}

    # If it exists, delete it
    cur.execute(f"DELETE FROM crime_data WHERE crime_id={data.crimeID}")
    conn.commit()

    cur.close()
    conn.close()

    return {"result": "Crime Data Deleted."}

class CrimeDataEdit(BaseModel):
    crimeID: int
    district: str
    barangay: str
    dateReported: str
    timeReported: str
    dateCommitted: str
    timeCommitted: str
    offense: str
    category: str
    latitude: str
    longitude: str
    weight: str
    year: str
    month: str
    time: str
    lightCondition: str
    day: str
    officer: str

@app.post("/edit")
async def edit(data: CrimeDataEdit):
    print(data.crimeID)
    conn = create_conn()
    cur = conn.cursor()

    # Check if the crime data exists
    cur.execute(f"SELECT * FROM crime_data WHERE crime_id={data.crimeID}")
    result = cur.fetchone()

    if result is None:
        return {"result": f"Crime data with the specified ID does not exist."}

    # If it exists, edit it
    cur.execute(f"UPDATE crime_data SET district = '{data.district}', barangay = '{data.barangay}', date_reported = '{data.dateReported}', \
        time_reported = '{data.timeReported}', date_committed = '{data.dateCommitted}', time_committed = '{data.timeCommitted}', \
        offenses = '{data.offense}', category = '{data.category}', latitude = '{data.latitude}', longitude = '{data.longitude}', \
        weights = '{data.weight}', year = '{data.year}', month = '{data.month}', time = '{data.time}', light_condition = '{data.lightCondition}', \
        day = '{data.day}', officer = '{data.officer}' WHERE crime_id={data.crimeID}")
    conn.commit()

    cur.close()
    conn.close()

    return {"result": "Crime Data Updated."}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
