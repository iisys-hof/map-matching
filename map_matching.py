#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cython: profile=True

#  Copyright (C) 2019-2021 Adrian Wöltche
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

import os
import time
import pickle

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


# load OSM network
print("Loading osm network...", end=" ")
if os.path.exists('data/network.pickle'):
    with open('data/network.pickle', 'rb') as handler:
        graph = pickle.load(handler)
    network = lib.network.Network("Hof, Bayern, Deutschland", graph=graph)
else:
    network = lib.network.Network("Hof, Bayern, Deutschland", result=1, simplify=True, strict=False)
    with open('data/network.pickle', 'wb') as handler:
        pickle.dump(network.graph, handler)
print("done")

# load Floating Car tracks
print("Loading floating car data...", end=" ")
tracks = lib.tracks.Tracks("data/points_anonymized.csv", delimiter=';', groupby=['device', 'subid'])
print("done")

plot = True
export = True

# learning
print("Building environment...", end=" ")
env = lib.learning.MapMatchingEnv(network, tracks)
print("done")

seed = env.seed(0)
print("Selecting track...", end=" ")
# Select a track via id and subid here
name = ('5c21e1e0-7651-4574-9b86-3aef14f6ecd5', 0)

track_points, track_line = tracks.get_track(name)
prepare = False
#prepare_buffer = (2*4.07, 10*4.07)
#prepare_buffer = (50, 200)
#prepare_buffer = (0.01, 0.01)
prepare_buffer = None
env.set_track(name, buffer=200, prepare=prepare, prepare_type='circle', prepare_buffer=prepare_buffer, state_steps=2, max_skip=3)
print("done")

graph = env.graph_updated
candidates = env.track_candidates
states_estimated = env.estimate_states()

print("Candidate combinations (estimate): {}".format(states_estimated))

network.export_track(graph, candidates, track_line, plot=plot, image_file="images/track.png", export=export)
network.export_candidates(env.graph_clean, candidates, plot=plot, image_file="images/candidates.png", export=export)
tracks.export_prepare(graph, track_points.copy(), prepare_type='triangle', prepare_buffer=prepare_buffer, plot=plot, image_file="images/prepare_triangle.png", export=export, export_file_pattern="exports/prepare_triangle_{}.geojson")
tracks.export_prepare(graph, track_points.copy(), prepare_type='circle', prepare_buffer=prepare_buffer, plot=plot, image_file="images/prepare_circle.png", export=export, export_file_pattern="exports/prepare_circle_{}.geojson")

learning = lib.learning.MapMatchingLearning()
episodes = 100 * states_estimated
improvement_break = 10 * len(track_points)

# value iteration
env.seed(seed[0])
env.reset_memory()
value_iteration_agent = benchmark("Value Iteration", lambda: learning.value_iteration(env, threshold=0.001, discount=1))
print("Calculated steps: {}".format(len(env.memory)))
value_iteration_route = network.route_line(graph, network.route_extract(graph, candidates, value_iteration_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, value_iteration_route, False))
learning.export_agent(value_iteration_agent, title="Value Iteration", plot=plot, image_file="images/value_iteration_score.png", export=export, export_file="exports/value_iteration_score.csv")
network.export_route(graph, candidates, value_iteration_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/value_iteration_policy.png", export=export, export_file_pattern="exports/value_iteration_policy_{}.geojson")
network.export_track_route(graph, candidates, value_iteration_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/value_iteration_track.png", export=False)

# nearest
env.seed(seed[0])
env.reset_memory()
nearest_agent = benchmark("Nearest", lambda: learning.nearest(env))
print("Calculated steps: {}".format(len(env.memory)))
nearest_route = network.route_line(graph, network.route_extract(graph, candidates, nearest_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, nearest_route, False))
learning.export_agent(nearest_agent, title="Nearest", plot=plot, image_file="images/nearest_score.png", export=export, export_file="exports/nearest_score.csv")
network.export_route(graph, candidates, nearest_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/nearest_policy.png", export=export, export_file_pattern="exports/nearest_policy_{}.geojson")
network.export_track_route(graph, candidates, nearest_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/nearest_track.png", export=False)

# greedy
env.seed(seed[0])
env.reset_memory()
greedy_agent = benchmark("Greedy", lambda: learning.greedy(env))
print("Calculated steps: {}".format(len(env.memory)))
greedy_route = network.route_line(graph, network.route_extract(graph, candidates, greedy_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, greedy_route, False))
learning.export_agent(greedy_agent, title="Greedy", plot=plot, image_file="images/greedy_score.png", export=export, export_file="exports/greedy_score.csv")
network.export_route(graph, candidates, greedy_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/greedy_policy.png", export=export, export_file_pattern="exports/greedy_policy_{}.geojson")
network.export_track_route(graph, candidates, greedy_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/greedy_track.png", export=False)

# qlearning epsilon intelligent
env.seed(seed[0])
env.reset_memory()
qlearning_intelligent_agent = benchmark("Q-Learning Epsilon Intelligent", lambda: learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
print("Calculated steps: {}".format(len(env.memory)))
qlearning_intelligent_route = network.route_line(graph, network.route_extract(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, qlearning_intelligent_route, False))
learning.export_agent(qlearning_intelligent_agent, title="Q-Learning Epsilon Intelligent", plot=plot, image_file="images/qlearning_epsilon_intelligent_score.png", export=export, export_file="exports/qlearning_epsilon_intelligent_score.csv")
network.export_route(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/qlearning_epsilon_intelligent_policy.png", export=export, export_file_pattern="exports/qlearning_epsilon_intelligent_policy_{}.geojson")
network.export_track_route(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/qlearning_epsilon_intelligent_track.png", export=False)

# qlearning epsilon intelligent inf
env.seed(seed[0])
env.reset_memory()
qlearning_intelligent_agent = benchmark("Q-Learning Epsilon Intelligent Inf", lambda: learning.qlearning(env, epsilon=0.5, samples=float('inf'), learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
print("Calculated steps: {}".format(len(env.memory)))
qlearning_intelligent_route = network.route_line(graph, network.route_extract(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, qlearning_intelligent_route, False))
learning.export_agent(qlearning_intelligent_agent, title="Q-Learning Epsilon Intelligent Inf", plot=plot, image_file="images/qlearning_epsilon_intelligent_inf_score.png", export=export, export_file="exports/qlearning_epsilon_intelligent_inf_score.csv")
network.export_route(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/qlearning_epsilon_intelligent_inf_policy.png", export=export, export_file_pattern="exports/qlearning_epsilon_intelligent_inf_policy_{}.geojson")
network.export_track_route(graph, candidates, qlearning_intelligent_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/qlearning_epsilon_intelligent_inf_track.png", export=False)

# qlearning epsilon greedy
env.seed(seed[0])
env.reset_memory()
qlearning_greedy_agent = benchmark("Q-Learning Epsilon Greedy", lambda: learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break, intelligent=False))
print("Calculated steps: {}".format(len(env.memory)))
qlearning_greedy_route = network.route_line(graph, network.route_extract(graph, candidates, qlearning_greedy_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, qlearning_greedy_route, False))
learning.export_agent(qlearning_greedy_agent, title="Q-Learning Greedy", plot=plot, image_file="images/qlearning_greedy_score.png", export=export, export_file="exports/qlearning_greedy_score.csv")
network.export_route(graph, candidates, qlearning_greedy_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/qlearning_greedy_policy.png", export=export, export_file_pattern="exports/qlearning_greedy_policy_{}.geojson")
network.export_track_route(graph, candidates, qlearning_greedy_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/qlearning_greedy_track.png", export=False)

# qlearning epsilon decay
env.seed(seed[0])
env.reset_memory()
qlearning_epsilon_decay_agent = benchmark("Q-Learning Epsilon Decay", lambda: learning._qlearning(env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
print("Calculated steps: {}".format(len(env.memory)))
qlearning_epsilon_decay_route = network.route_line(graph, network.route_extract(graph, candidates, qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, qlearning_epsilon_decay_route, False))
learning.export_agent(qlearning_epsilon_decay_agent, title="Q-Learning Epsilon", plot=plot, image_file="images/qlearning_epsilon_decay_score.png", export=export, export_file="exports/qlearning_epsilon_decay_score.csv")
network.export_route(graph, candidates, qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/qlearning_epsilon_decay_policy.png", export=export, export_file_pattern="exports/qlearning_epsilon_decay_policy_{}.geojson")
network.export_track_route(graph, candidates, qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/qlearning_epsilon_decay_track.png", export=False)

# qlearning epsilon
env.seed(seed[0])
env.reset_memory()
qlearning_epsilon_agent = benchmark("Q-Learning Epsilon", lambda: learning._qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
print("Calculated steps: {}".format(len(env.memory)))
qlearning_epsilon_route = network.route_line(graph, network.route_extract(graph, candidates, qlearning_epsilon_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, qlearning_epsilon_route, False))
learning.export_agent(qlearning_epsilon_agent, title="Q-Learning", plot=plot, image_file="images/qlearning_epsilon_score.png", export=export, export_file="exports/qlearning_epsilon_score.csv")
network.export_route(graph, candidates, qlearning_epsilon_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/qlearning_epsilon_policy.png", export=export, export_file_pattern="exports/qlearning_epsilon_policy_{}.geojson")
network.export_track_route(graph, candidates, qlearning_epsilon_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/qlearning_epsilon_track.png", export=False)

# # double_qlearning epsilon decay
# env.seed(seed[0])
# env.reset_memory()
# double_qlearning_epsilon_decay_agent = benchmark("Double Q-Learning Epsilon Decay", lambda: learning._double_qlearning(env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, learning_rate=0.2, discount=1, episodes=episodes, improvement_break=improvement_break))
# print("Calculated steps: {}".format(len(env.memory)))
# double_qlearning_epsilon_decay_route = network.route_line(graph, network.route_extract(graph, candidates, double_qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes))['geometry'][0]
# print("Comparison: ", tracks.compare_matches(value_iteration_route, double_qlearning_epsilon_decay_route, False))
# learning.export_agent(double_qlearning_epsilon_decay_agent, title="Double Q-Learning Epsilon", plot=plot, image_file="images/double_qlearning_epsilon_decay_score.png", export=export, export_file="exports/double_qlearning_epsilon_decay_score.csv")
# network.export_route(graph, candidates, double_qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/double_qlearning_epsilon_decay_policy.png", export=export, export_file_pattern="exports/double_qlearning_epsilon_decay_policy_{}.geojson")
# network.export_track_route(graph, candidates, double_qlearning_epsilon_decay_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/double_qlearning_epsilon_decay_track.png", export=False)
#
# # double_qlearning epsilon
# env.seed(seed[0])
# env.reset_memory()
# double_qlearning_epsilon_agent = benchmark("Double Q-Learning Epsilon", lambda: learning._double_qlearning(env, epsilon=0.5, learning_rate=0.2, discount=1, episodes=episodes, improvement_break=improvement_break))
# print("Calculated steps: {}".format(len(env.memory)))
# double_qlearning_epsilon_route = network.route_line(graph, network.route_extract(graph, candidates, double_qlearning_epsilon_agent['policy'], routes_cache=env.routes))['geometry'][0]
# print("Comparison: ", tracks.compare_matches(value_iteration_route, double_qlearning_epsilon_route, False))
# learning.export_agent(double_qlearning_epsilon_agent, title="Double Q-Learning", plot=plot, image_file="images/double_qlearning_epsilon_score.png", export=export, export_file="exports/double_qlearning_epsilon_score.csv")
# network.export_route(graph, candidates, double_qlearning_epsilon_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/double_qlearning_epsilon_policy.png", export=export, export_file_pattern="exports/double_qlearning_epsilon_policy_{}.geojson")
# network.export_track_route(graph, candidates, double_qlearning_epsilon_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/double_qlearning_epsilon_track.png", export=False)

# # sarsa epsilon decay
# env.seed(seed[0])
# env.reset_memory()
# sarsa_epsilon_decay_agent = benchmark("SARSA Epsilon Decay", lambda: learning._sarsa(env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, learning_rate=0.1, discount=1, episodes=episodes, improvement_break=improvement_break))
# print("Calculated steps: {}".format(len(env.memory)))
# sarsa_epsilon_decay_route = network.route_line(graph, network.route_extract(graph, candidates, sarsa_epsilon_decay_agent['policy'], routes_cache=env.routes))['geometry'][0]
# print("Comparison: ", tracks.compare_matches(value_iteration_route, sarsa_epsilon_decay_route, False))
# learning.export_agent(sarsa_epsilon_decay_agent, title="SARSA Epsilon", plot=plot, image_file="images/sarsa_epsilon_decay_score.png", export=export, export_file="exports/sarsa_epsilon_decay_score.csv")
# network.export_route(graph, candidates, sarsa_epsilon_decay_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/sarsa_epsilon_decay_policy.png", export=export, export_file_pattern="exports/sarsa_epsilon_decay_policy_{}.geojson")
# network.export_track_route(graph, candidates, sarsa_epsilon_decay_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/sarsa_epsilon_decay_track.png", export=False)
#
# # sarsa epsilon
# env.seed(seed[0])
# env.reset_memory()
# sarsa_epsilon_agent = benchmark("SARSA Epsilon", lambda: learning._sarsa(env, epsilon=0.5, learning_rate=0.1, discount=1, episodes=episodes, improvement_break=improvement_break))
# print("Calculated steps: {}".format(len(env.memory)))
# sarsa_epsilon_route = network.route_line(graph, network.route_extract(graph, candidates, sarsa_epsilon_agent['policy'], routes_cache=env.routes))['geometry'][0]
# print("Comparison: ", tracks.compare_matches(value_iteration_route, sarsa_epsilon_route, False))
# learning.export_agent(sarsa_epsilon_agent, title="SARSA", plot=plot, image_file="images/sarsa_epsilon_score.png", export=export, export_file="exports/sarsa_epsilon_score.csv")
# network.export_route(graph, candidates, sarsa_epsilon_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/sarsa_epsilon_policy.png", export=export, export_file_pattern="exports/sarsa_epsilon_policy_{}.geojson")
# network.export_track_route(graph, candidates, sarsa_epsilon_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/sarsa_epsilon_track.png", export=False)

# expected sarsa epsilon decay
env.seed(seed[0])
env.reset_memory()
expected_sarsa_epsilon_decay_agent = benchmark("Expected SARSA Epsilon Decay", lambda: learning._expected_sarsa(env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
print("Calculated steps: {}".format(len(env.memory)))
expected_sarsa_epsilon_decay_route = network.route_line(graph, network.route_extract(graph, candidates, expected_sarsa_epsilon_decay_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, expected_sarsa_epsilon_decay_route, False))
learning.export_agent(expected_sarsa_epsilon_decay_agent, title="Expected SARSA Epsilon", plot=plot, image_file="images/expected_sarsa_epsilon_decay_score.png", export=export, export_file="exports/expected_sarsa_epsilon_decay_score.csv")
network.export_route(graph, candidates, expected_sarsa_epsilon_decay_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/expected_sarsa_epsilon_decay_policy.png", export=export, export_file_pattern="exports/expected_sarsa_epsilon_decay_policy_{}.geojson")
network.export_track_route(graph, candidates, expected_sarsa_epsilon_decay_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/expected_sarsa_epsilon_decay_track.png", export=False)

# expected sarsa epsilon
env.seed(seed[0])
env.reset_memory()
expected_sarsa_epsilon_agent = benchmark("Expected SARSA Epsilon", lambda: learning._expected_sarsa(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
print("Calculated steps: {}".format(len(env.memory)))
expected_sarsa_epsilon_route = network.route_line(graph, network.route_extract(graph, candidates, expected_sarsa_epsilon_agent['policy'], routes_cache=env.routes))['geometry'][0]
print("Comparison: ", tracks.compare_matches(value_iteration_route, expected_sarsa_epsilon_route, False))
learning.export_agent(expected_sarsa_epsilon_agent, title="Expected SARSA", plot=plot, image_file="images/expected_sarsa_epsilon_score.png", export=export, export_file="exports/expected_sarsa_epsilon_score.csv")
network.export_route(graph, candidates, expected_sarsa_epsilon_agent['policy'], routes_cache=env.routes, plot=plot, image_file="images/expected_sarsa_epsilon_policy.png", export=export, export_file_pattern="exports/expected_sarsa_epsilon_policy_{}.geojson")
network.export_track_route(graph, candidates, expected_sarsa_epsilon_agent['policy'], routes_cache=env.routes, track=track_line, plot=plot, image_file="images/expected_sarsa_epsilon_track.png", export=False)
