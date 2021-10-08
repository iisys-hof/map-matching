#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cython: profile=True

#  Copyright (C) 2020-2021 Adrian Wöltche
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program. If not, see https://www.gnu.org/licenses/.

import pickle
import os
import time

import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely as shp
import matplotlib.pyplot as plt

import lib.network
import lib.tracks
import lib.learning

def benchmark(name, function):
    print("Benchmark {}...".format(name), end=" ")
    start = time.time()
    value = function()
    end = time.time()
    print("{:.9f} seconds".format(end - start))
    return value

road_network_file = "data/ground_truth/newson_krumm/road_network.txt"
road_network_graph = "data/ground_truth/newson_krumm/road_network.graphml"
road_network_pickle = "data/ground_truth/newson_krumm/road_network.pickle"
gps_data_file = "data/ground_truth/newson_krumm/gps_data.txt"
gps_data_pickle = "data/ground_truth/newson_krumm/gps_data.pickle"
ground_truth_route_file = "data/ground_truth/newson_krumm/ground_truth_route.txt"
ground_truth_route_pickle = "data/ground_truth/newson_krumm/ground_truth_route.pickle"
crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
reproject_crs = "+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +unit"

def line_reverse(row):
    tmp = row["To Node ID"]
    row["To Node ID"] = row["From Node ID"]
    row["From Node ID"] = tmp
    reverse = list(row["LINESTRING()"].coords)
    reverse.reverse()
    row["LINESTRING()"] = shp.geometry.LineString(reverse)
    return row

# load OSM network
if os.path.exists(road_network_pickle):
    print("Loading graph network from pickle...", end=" ")
    with open(road_network_pickle, 'rb') as handler:
        graph = pickle.load(handler)
    network = lib.network.Network("Newson-Krumm", graph=graph)
    print("done")
else:
    if not os.path.exists(road_network_graph):
        print("Parsing road_network.txt...", end=" ")
        road_network_csv = pd.read_csv(road_network_file, delimiter='\t')
        road_network_csv = road_network_csv.set_index("Edge ID")
        road_network_csv["LINESTRING()"] = road_network_csv["LINESTRING()"].apply(shp.wkt.loads)
        print("done")
        
        print("Extracing edges...", end=" ")
        road_network_edges_original = gpd.GeoDataFrame(road_network_csv.drop(columns=["Vertex Count"]), crs=crs, geometry="LINESTRING()")
        road_network_edges_reversed = road_network_edges_original[road_network_edges_original["Two Way"] == 1].apply(line_reverse, axis=1)
        road_network_edges = pd.concat([road_network_edges_original, road_network_edges_reversed], axis=0, ignore_index=True).drop(columns=["Two Way"])
        print("done")
        
        print("Extracing nodes...", end=" ")
        road_network_nodes_from = gpd.GeoDataFrame(road_network_edges.drop(columns=["To Node ID", " Speed (m/s)", "LINESTRING()"]), crs=crs,
                                                   geometry=gpd.points_from_xy(road_network_edges["LINESTRING()"].apply(lambda line: line.coords[0][0]),
                                                                               road_network_edges["LINESTRING()"].apply(lambda line: line.coords[0][1])))
        
        road_network_nodes_from = road_network_nodes_from.set_index("From Node ID")
        road_network_nodes_to = gpd.GeoDataFrame(road_network_edges.drop(columns=["From Node ID", " Speed (m/s)", "LINESTRING()"]), crs=crs,
                                                 geometry=gpd.points_from_xy(road_network_edges["LINESTRING()"].apply(lambda line: line.coords[-1][0]),
                                                                             road_network_edges["LINESTRING()"].apply(lambda line: line.coords[-1][1])))
        road_network_nodes_to = road_network_nodes_to.set_index("To Node ID")
        
        road_network_nodes = pd.concat([road_network_nodes_from, road_network_nodes_to])
        road_network_nodes = road_network_nodes.loc[~road_network_nodes.index.duplicated()]
        print("done")
        
        print("Replacing duplicate nodes...", end=" ")
        road_network_nodes_duplicated = road_network_nodes.loc[road_network_nodes.duplicated(["geometry"], keep=False)]
        
        duplicate_nodes = {}
        for name, data in road_network_nodes_duplicated.iterrows():
            wkt = data["geometry"].to_wkt()
            if wkt not in duplicate_nodes:
                duplicate_nodes[wkt] = []
            duplicate_nodes[wkt].append(name)
        
        duplicate_drop = []
        for wkt in duplicate_nodes:
            nodes = duplicate_nodes[wkt]
            replace_node = nodes[0]
            to_replace = nodes[1:]
            duplicate_drop.extend(to_replace)
            
            road_network_edges = road_network_edges.replace({"From Node ID": to_replace, "To Node ID": to_replace}, replace_node)
        road_network_nodes = road_network_nodes.drop(duplicate_drop)
        print("done")
        
        print("Converting to graph...", end=" ")
        road_network_nodes["osmid"] = road_network_nodes.index
        road_network_nodes["x"] = road_network_nodes["geometry"].apply(lambda point: point.x)
        road_network_nodes["y"] = road_network_nodes["geometry"].apply(lambda point: point.y)
        road_network_nodes.gdf_name = "newson_krumm_nodes"
        road_network_edges = road_network_edges.rename(columns={"From Node ID": "u", "To Node ID": "v", " Speed (m/s)": "speed", "LINESTRING()": "geometry"})
        road_network_edges["key"] = 0
        road_network_edges["oneway"] = True
        
        graph = ox.utils_graph.graph_from_gdfs(road_network_nodes, road_network_edges)
        graph = ox.utils_graph.add_edge_lengths(graph)
        print("done")
        
        print("Saving graph...", end=" ")
        ox.save_graphml(graph, filepath=road_network_graph)
        print("done")

    print("Loading graph network...", end=" ")
    network = lib.network.Network("Newson-Krumm", file=road_network_graph)
    graph = network.graph
    print("done")
    
    print("Saving graph network to pickle...", end=" ")
    with open(road_network_pickle, 'wb') as handler:
        pickle.dump(graph, handler)
    print("done")

# load GPS data track
if os.path.exists(gps_data_pickle):
    print("Loading GPS data from pickle...", end=" ")
    with open(gps_data_pickle, 'rb') as handler:
        points = pickle.load(handler)
    tracks = lib.tracks.Tracks(points=points, groupby=['device'])
    print("done")
else:
    print("Parsing gps_data.txt...", end=" ")
    gps_data_csv = pd.read_csv(gps_data_file, delimiter='\t', parse_dates=[["Date (UTC)", "Time (UTC)"]], date_parser=lambda col: pd.to_datetime(col, utc=True))
    gps_data_points = gpd.GeoDataFrame(gps_data_csv.drop(columns=["Latitude", "Longitude", "Unnamed: 4", "Unnamed: 5"]), crs=crs, 
                                       geometry=gpd.points_from_xy(gps_data_csv["Longitude"], gps_data_csv["Latitude"]))
    gps_data_points = gps_data_points.to_crs(reproject_crs)
    
    gps_data_points = gps_data_points.rename(columns={"Date (UTC)_Time (UTC)": "timestamp"})
    gps_data_points["device"] = "newson_krumm"
    print("done")

    print("Loading GPS data...", end=" ")
    tracks = lib.tracks.Tracks(points=gps_data_points, groupby=['device'])
    points = tracks.points
    print("done")
    
    print("Saving GPS data to pickle...", end=" ")
    with open(gps_data_pickle, 'wb') as handler:
        pickle.dump(points, handler)
    print("done")

# load ground truth route
if os.path.exists(ground_truth_route_pickle):
    print("Loading ground truth route from pickle...", end=" ")
    with open(ground_truth_route_pickle, 'rb') as handler:
        ground_truth_route_line = pickle.load(handler)
    print("done")
else:
    print("Parsing ground_truth_route.txt...", end=" ")
    ground_truth_route_csv = pd.read_csv(ground_truth_route_file, delimiter='\t')
    ground_truth_route_csv = ground_truth_route_csv.set_index("Edge ID")
    print("done")
    
    print("Parsing road_network.txt...", end=" ")
    road_network_csv = pd.read_csv(road_network_file, delimiter='\t')
    road_network_csv = road_network_csv.set_index("Edge ID")
    road_network_csv["LINESTRING()"] = road_network_csv["LINESTRING()"].apply(shp.wkt.loads)
    print("done")
    
    print("Extracting ground truth route...", end=" ")
    ground_truth_network_list = []
    for edge_id, ground_truth_route in ground_truth_route_csv.iterrows():
        ground_truth_network_list.append(ground_truth_route.append(road_network_csv.loc[edge_id]))
    ground_truth_network_edges = gpd.GeoDataFrame(ground_truth_network_list, geometry="LINESTRING()", crs=crs)
    ground_truth_network_edges = ground_truth_network_edges.to_crs(reproject_crs)
    ground_truth_network_edges = ground_truth_network_edges.drop(columns=["Vertex Count"])
    ground_truth_network_edges_reversed = ground_truth_network_edges[ground_truth_network_edges["Traversed From to To"] == 0].apply(line_reverse, axis=1)
    ground_truth_network_edges.update(ground_truth_network_edges_reversed)
    ground_truth_network_edges = ground_truth_network_edges.rename(columns={"Traversed From to To": "direction", "From Node ID": "u", "To Node ID": "v", " Speed (m/s)": "speed", "Two Way": "two_way", "LINESTRING()": "geometry"})
    print("done")
    
    print("Creating ground truth route line...", end=" ")
    ground_truth_route_lines = ground_truth_network_edges["geometry"]
    ground_truth_route_coords = [list(line.coords) for line in ground_truth_route_lines]
    ground_truth_route_coords_line = []
    for i in range(len(ground_truth_route_coords)):
        if i+1 < len(ground_truth_route_coords) and ground_truth_route_coords[i][-1] == ground_truth_route_coords[i+1][0]:
            ground_truth_route_coords_line.extend(ground_truth_route_coords[i][:-1])
        else:
            ground_truth_route_coords_line.extend(ground_truth_route_coords[i])
    ground_truth_route_line = gpd.GeoDataFrame(geometry=[shp.geometry.LineString(ground_truth_route_coords_line)], crs=reproject_crs)
    print("done")
    
    print("Saving ground truth route to pickle...", end=" ")
    with open(ground_truth_route_pickle, 'wb') as handler:
        pickle.dump(ground_truth_route_line, handler)
    print("done")

# cut points
# tracks.points = tracks.points[1200:1300]
# tracks.points = tracks.points[1215:1245]
tracks.points_group = tracks.points.groupby(['device'])

plot = True
export = True

if plot and not os.path.exists('images/newson_krumm'):
    os.mkdir('images/newson_krumm')
if export and not os.path.exists('exports/newson_krumm'):
    os.mkdir('exports/newson_krumm')

# learning
print("Building environment...", end=" ")
env = lib.learning.MapMatchingEnv(network, tracks)
seed = env.seed(0)
print("done")

print("Selecting track...", end=" ")
name = 'newson_krumm'
ground_truth_route = ground_truth_route_line["geometry"][0]
ground_truth_route_line.to_file('exports/newson_krumm/ground_truth_route.geojson', driver='GeoJSON')

track_points, track_line = tracks.get_track(name)
prepare_buffer = (2*4.07, 10*4.07)
env.set_track(name, buffer=200, prepare_type='circle', prepare_buffer=prepare_buffer, state_steps=2)
print("done")

graph = env.graph_updated
candidates = env.track_candidates
states_estimated = env.estimate_states()

print("Candidate combinations (estimate): {}".format(states_estimated))

# if plot:
#     ox.plot_graph(network.graph, node_size=0.05, edge_linewidth=0.05, filename='newson_krumm/route_network', dpi=1200, save=True, show=False, close=True)
#     ox.plot_graph(env.graph_clean, node_size=0.1, edge_linewidth=0.1, filename='newson_krumm/route_network_extract', dpi=1200, save=True, show=False, close=True)

network.export_track(graph, candidates, track_line, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/track.png", export=export, export_file_pattern="exports/newson_krumm/track_{}.geojson")
network.export_candidates(env.graph_clean, candidates, plot=plot, dpi=1200, node_size=0.05, edge_linewidth=0.1, image_file="images/newson_krumm/candidates.png", export=export, export_file_pattern="exports/newson_krumm/candidates_{}.geojson")
tracks.export_prepare(graph, track_points.copy(), prepare_type='triangle', prepare_buffer=prepare_buffer, plot=plot, dpi=1200, node_size=0.05, edge_linewidth=0.1, image_file="images/newson_krumm/prepare_triangle.png", export=export, export_file_pattern="exports/newson_krumm/prepare_triangle_{}.geojson")
tracks.export_prepare(graph, track_points.copy(), prepare_type='circle', prepare_buffer=prepare_buffer, plot=plot, dpi=1200, node_size=0.05, edge_linewidth=0.1, image_file="images/newson_krumm/prepare_circle.png", export=export, export_file_pattern="exports/newson_krumm/prepare_circle_{}.geojson")

learning = lib.learning.MapMatchingLearning()
episodes = 10 * states_estimated
improvement_break = 5 * len(track_points)

# value iteration
env.seed(seed[0])
env.reset_memory()
value_iteration_agent = benchmark("Value Iteration", lambda: learning.value_iteration(env, threshold=0.001, discount=1))
# value_iteration_agent = learning.value_iteration(env, threshold=0.001, discount=1, prints=True)
print("Calculated steps: {}".format(len(env.memory)))
value_iteration_route = network.route_line(graph, network.route_extract(graph, candidates, value_iteration_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, value_iteration_route, True))
print("Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, value_iteration_route, False))
learning.export_agent(value_iteration_agent, title="Value Iteration", plot=plot, image_file="images/newson_krumm/value_iteration_score.png", export=export, export_file="exports/newson_krumm/value_iteration_score.csv")
network.export_route(graph, candidates, value_iteration_agent['policy'], routes_cache=env.routes, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/value_iteration_policy.png", export=export, export_file_pattern="exports/newson_krumm/value_iteration_policy_{}.geojson")
network.export_track_route(graph, candidates, value_iteration_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/value_iteration_track.png", export=False)

# nearest
env.seed(seed[0])
env.reset_memory()
nearest_agent = benchmark("Nearest", lambda: learning.nearest(env))
# nearest_agent = learning.nearest(env, prints=True)
print("Calculated steps: {}".format(len(env.memory)))
nearest_route = network.route_line(graph, network.route_extract(graph, candidates, nearest_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, nearest_route, True))
print("Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, nearest_route, False))
learning.export_agent(nearest_agent, title="Nearest", plot=plot, image_file="images/newson_krumm/nearest_score.png", export=export, export_file="exports/newson_krumm/nearest_score.csv")
network.export_route(graph, candidates, nearest_agent['policy'], routes_cache=env.routes, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/nearest_policy.png", export=export, export_file_pattern="exports/newson_krumm/nearest_policy_{}.geojson")
network.export_track_route(graph, candidates, nearest_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/nearest_track.png", export=False)

# greedy
env.seed(seed[0])
env.reset_memory()
greedy_agent = benchmark("Greedy", lambda: learning.greedy(env))
# greedy_agent = learning.greedy(env, prints=True)
print("Calculated steps: {}".format(len(env.memory)))
greedy_route = network.route_line(graph, network.route_extract(graph, candidates, greedy_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, greedy_route, True))
print("Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, greedy_route, False))
learning.export_agent(greedy_agent, title="Greedy", plot=plot, image_file="images/newson_krumm/greedy_score.png", export=export, export_file="exports/newson_krumm/greedy_score.csv")
network.export_route(graph, candidates, greedy_agent['policy'], routes_cache=env.routes, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/greedy_policy.png", export=export, export_file_pattern="exports/newson_krumm/greedy_policy_{}.geojson")
network.export_track_route(graph, candidates, greedy_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/greedy_track.png", export=False)

# qlearning epsilon intelligent
env.seed(seed[0])
env.reset_memory()
qlearning_intelligent_agent = benchmark("Q-Learning Epsilon Intelligent", lambda: learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
# qlearning_intelligent_agent = learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break, prints=True)
print("Calculated steps: {}".format(len(env.memory)))
qlearning_epsilon_route = network.route_line(graph, network.route_extract(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_route, True))
print("Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_route, False))
learning.export_agent(qlearning_intelligent_agent, title="Q-Learning Intelligent", plot=plot, image_file="images/newson_krumm/qlearning_intelligent_score.png", export=export, export_file="exports/newson_krumm/qlearning_intelligent_score.csv")
network.export_route(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/qlearning_intelligent_policy.png", export=export, export_file_pattern="exports/newson_krumm/qlearning_intelligent_policy_{}.geojson")
network.export_track_route(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/qlearning_intelligent_track.png", export=False)

# qlearning epsilon decay
env.seed(seed[0])
env.reset_memory()
qlearning_epsilon_decay_agent = benchmark("Q-Learning Epsilon Decay", lambda: learning._qlearning(env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
# qlearning_epsilon_decay_agent = learning.qlearning(env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break, prints=True)
print("Calculated steps: {}".format(len(env.memory)))
qlearning_epsilon_route = network.route_line(graph, network.route_extract(graph, candidates, qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_route, True))
print("Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_route, False))
learning.export_agent(qlearning_epsilon_decay_agent, title="Q-Learning Epsilon", plot=plot, image_file="images/newson_krumm/qlearning_epsilon_decay_score.png", export=export, export_file="exports/newson_krumm/qlearning_epsilon_decay_score.csv")
network.export_route(graph, candidates, qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/qlearning_epsilon_decay_policy.png", export=export, export_file_pattern="exports/newson_krumm/qlearning_epsilon_decay_policy_{}.geojson")
network.export_track_route(graph, candidates, qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/qlearning_epsilon_decay_track.png", export=False)

# qlearning epsilon
env.seed(seed[0])
env.reset_memory()
qlearning_epsilon_agent = benchmark("Q-Learning Epsilon", lambda: learning._qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
# qlearning_epsilon_agent = learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break, prints=True)
print("Calculated steps: {}".format(len(env.memory)))
qlearning_epsilon_route = network.route_line(graph, network.route_extract(graph, candidates, qlearning_epsilon_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_route, True))
print("Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_route, False))
learning.export_agent(qlearning_epsilon_agent, title="Q-Learning", plot=plot, image_file="images/newson_krumm/qlearning_epsilon_score.png", export=export, export_file="exports/newson_krumm/qlearning_epsilon_score.csv")
network.export_route(graph, candidates, qlearning_epsilon_agent['policy'], routes_cache=env.routes, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/qlearning_epsilon_policy.png", export=export, export_file_pattern="exports/newson_krumm/qlearning_epsilon_policy_{}.geojson")
network.export_track_route(graph, candidates, qlearning_epsilon_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, dpi=1200, node_size=0.01, edge_linewidth=0.1, image_file="images/newson_krumm/qlearning_epsilon_track.png", export=False)
