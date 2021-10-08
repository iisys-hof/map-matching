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

import geopandas as gpd

import lib.geodata


def benchmark(name, function):
    print("Benchmark {}...".format(name), end=" ")
    start = time.time()
    value = function()
    end = time.time()
    print("{:.9f} seconds".format(end - start))
    return value


ground_truth_route_pickle = "data/ground_truth/newson_krumm/ground_truth_route.pickle"
crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
reproject_crs = "+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +unit"

# result folder
result_folder = 'exports/newson_krumm_01/'

tracks = lib.geodata.GeoData()

# load ground truth route
print("Loading ground truth route from pickle...", end=" ")
with open(ground_truth_route_pickle, 'rb') as handler:
    ground_truth_route_line = pickle.load(handler)
print("done")
ground_truth_route = ground_truth_route_line["geometry"][0]

# value iteration
value_iteration_route = gpd.read_file(os.path.join(result_folder, 'value_iteration_policy_line.geojson'))["geometry"][0]
print("Value Iteration Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, value_iteration_route, True))
print("Value Iteration Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, value_iteration_route, False))

# # nearest
# nearest_route = gpd.read_file(os.path.join(result_folder, 'nearest_policy_line.geojson'))["geometry"][0]
# print("Nearest Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, nearest_route, True))
# print("Nearest Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, nearest_route, False))
#
# # greedy
# greedy_route = gpd.read_file(os.path.join(result_folder, 'greedy_policy_line.geojson'))["geometry"][0]
# print("Greedy Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, greedy_route, True))
# print("Greedy Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, greedy_route, False))
#
# # qlearning epsilon intelligent
# qlearning_intelligent_route = gpd.read_file(os.path.join(result_folder, 'qlearning_intelligent_policy_line.geojson'))["geometry"][0]
# print("Q-Learning Intelligent Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, qlearning_intelligent_route, True))
# print("Q-Learning Intelligent Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, qlearning_intelligent_route, False))
#
# # qlearning epsilon decay
# qlearning_epsilon_decay_route = gpd.read_file(os.path.join(result_folder, 'qlearning_epsilon_decay_policy_line.geojson'))["geometry"][0]
# print("Q-Learning Epsilon Decay Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_decay_route, True))
# print("Q-Learning Epsilon Decay Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_decay_route, False))
#
# # qlearning epsilon
# qlearning_epsilon_route = gpd.read_file(os.path.join(result_folder, 'qlearning_epsilon_policy_line.geojson'))["geometry"][0]
# print("Q-Learning Epsilon Comparison (DoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_route, True))
# print("Q-Learning Epsilon Comparison (NoCut): ", tracks.compare_matches(ground_truth_route, qlearning_epsilon_route, False))
