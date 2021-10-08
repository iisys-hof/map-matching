#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import sys
import os
import time
import pickle

import pandas as pd

import lib.network
import lib.tracks
import lib.learning


def benchmark(function):
    start = time.time()
    value = function()
    end = time.time()
    return value, end - start


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

buffer = 100
steps = 2

results = {}
points = 0

for name in tracks.points_group.groups.keys():
    optimal_route = None
    optimal_states = None

    env = lib.learning.MapMatchingEnv(network, tracks)
    env.seed(0)  # for comparability
    learning = lib.learning.MapMatchingLearning()

    print("")
    print("Selecting track {}".format(name))

    for algorithm in ["value_iteration", "nearest", "greedy", "qlearning_intelligent", "qlearning_greedy", "qlearning_epsilon_decay", "qlearning_epsilon", "expected_sarsa_decay", "expected_sarsa"]:
        try:
            print("Evaluating {} ...".format(algorithm.rjust(25)), end=" ")

            env.set_track(name, buffer=buffer, state_steps=steps)
            points += len(env.track_points)

            if len(env.track_candidates) <= 2:
                print("Track too short (length: {}), skipping...".format(len(env.track_candidates)))
                break

            states_estimated = env.estimate_states()

            episodes = 100 * states_estimated
            improvement_break = 10 * len(env.track_points)

            agent = None
            duration = 0.0
            if algorithm == "value_iteration":
                agent, duration = benchmark(lambda: learning.value_iteration(env, threshold=0.001, discount=1))
            elif algorithm == "nearest":
                agent, duration = benchmark(lambda: learning.nearest(env))
            elif algorithm == "greedy":
                agent, duration = benchmark(lambda: learning.greedy(env))
            elif algorithm == "qlearning_intelligent":
                agent, duration = benchmark(lambda: learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break, intelligent=True))
            elif algorithm == "qlearning_greedy":
                agent, duration = benchmark(lambda: learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break, intelligent=False))
            elif algorithm == "qlearning_epsilon_decay":
                agent, duration = benchmark(lambda: learning._qlearning(env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
            elif algorithm == "qlearning_epsilon":
                agent, duration = benchmark(lambda: learning._qlearning(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
            elif algorithm == "expected_sarsa_decay":
                agent, duration = benchmark(lambda: learning._expected_sarsa(env, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))
            elif algorithm == "expected_sarsa":
                agent, duration = benchmark(lambda: learning._expected_sarsa(env, epsilon=0.5, learning_rate=1.0, discount=1, episodes=episodes, improvement_break=improvement_break))

            states_calculated = len(env.memory)
            route = network.route_line(env.graph_updated,
                                       network.route_extract(env.graph_updated, env.track_candidates, agent['policy'],
                                                             routes_cache=env.routes))['geometry'][0]

            if algorithm == "value_iteration":
                optimal_route = route
                optimal_states = states_calculated

            # error_add, error_miss, error_fraction = tracks.compare_matches(optimal_route, route, False)
            error_add, error_miss, error_fraction = (0, 0, 0)
            states_percentage = states_calculated / optimal_states

            print(
                "done, {:9.3f} seconds, {:8} states calculated, {:6.2f} % states, {:9.2f} error added, {:9.2f} error missed, {:6.2f} error fraction".format(
                    duration, states_calculated, states_percentage * 100, error_add, error_miss, error_fraction))

            if name not in results:
                results[name] = {}
            results[name]["{}_{}".format(algorithm, 'duration')] = duration
            results[name]["{}_{}".format(algorithm, 'states_calculated')] = states_calculated
            results[name]["{}_{}".format(algorithm, 'states_percentage')] = states_percentage
            results[name]["{}_{}".format(algorithm, 'error_added')] = error_add
            results[name]["{}_{}".format(algorithm, 'error_missed')] = error_miss
            results[name]["{}_{}".format(algorithm, 'error_fraction')] = error_fraction

        except KeyboardInterrupt:
            exit(1)
        except:
            print("aborted, unexpected error:", sys.exc_info())
print("")
print("Finished.")
print("Points:", points)

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv("benchmark_all_cache.csv")
