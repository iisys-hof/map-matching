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

buffer = 200
steps = 2

results = {}

for name in tracks.points_group.groups.keys():
    optimal_route = None
    optimal_states = None

    print("")
    print("Selecting track {}".format(name))

    too_short_quit = False
    for algorithm in ["value_iteration", "qlearning_intelligent", "qlearning_greedy"]:
        if too_short_quit:
            too_short_quit = False
            break

        for gamma in [1.0, 0.8, 0.6, 0.4, 0.2]:
            try:
                print("Evaluating {} with discount {} ...".format(algorithm.rjust(25), gamma), end=" ")

                env = lib.learning.MapMatchingEnv(network, tracks)
                env.seed(0)  # for comparability
                learning = lib.learning.MapMatchingLearning()

                track_points, track_line = tracks.get_track(name)
                env.set_track(name, buffer=buffer, state_steps=steps)

                if len(env.track_candidates) <= 2:
                    print("Track too short (length: {}), skipping...".format(len(env.track_candidates)))
                    too_short_quit = True
                    break

                states_estimated = env.estimate_states()

                episodes = 100 * states_estimated
                improvement_break = 10 * len(track_points)

                agent = None
                duration = 0.0
                if algorithm == "value_iteration":
                    agent, duration = benchmark(lambda: learning.value_iteration(env, threshold=0.001, discount=gamma))
                elif algorithm == "qlearning_intelligent":
                    agent, duration = benchmark(lambda: learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=gamma, episodes=episodes, improvement_break=improvement_break, intelligent=True))
                elif algorithm == "qlearning_greedy":
                    agent, duration = benchmark(lambda: learning.qlearning(env, epsilon=0.5, learning_rate=1.0, discount=gamma, episodes=episodes, improvement_break=improvement_break, intelligent=False))

                states_calculated = len(env.memory)
                route = network.route_line(env.graph_updated, network.route_extract(env.graph_updated, env.track_candidates, agent['policy'], routes_cache=env.routes))['geometry'][0]

                if algorithm == "value_iteration" and gamma == 1.0:
                    optimal_route = route
                    optimal_states = states_calculated

                error_add, error_miss, error_fraction = tracks.compare_matches(optimal_route, route, False)
                states_percentage = states_calculated / optimal_states

                print("done, {} gamma, {:9.3f} seconds, {:8} states calculated, {:6.2f} % states, {:9.2f} error added, {:9.2f} error missed, {:6.2f} error fraction".format(
                    gamma, duration, states_calculated, states_percentage * 100, error_add, error_miss, error_fraction))

                if name not in results:
                    results[name] = {}
                results[name]["{}_{}_{}".format(algorithm, gamma, 'duration')] = duration
                results[name]["{}_{}_{}".format(algorithm, gamma, 'states_calculated')] = states_calculated
                results[name]["{}_{}_{}".format(algorithm, gamma, 'states_percentage')] = states_percentage
                results[name]["{}_{}_{}".format(algorithm, gamma, 'error_added')] = error_add
                results[name]["{}_{}_{}".format(algorithm, gamma, 'error_missed')] = error_miss
                results[name]["{}_{}_{}".format(algorithm, gamma, 'error_fraction')] = error_fraction

            except KeyboardInterrupt:
                exit(1)
            except:
                print("aborted, unexpected error:", sys.exc_info())

print("")
print("Finished.")

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv("benchmark_gamma.csv")
