# -*- coding: utf-8 -*-

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
import collections
import math

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lib.network
import lib.tracks

class VaryingDiscreteSpace(gym.spaces.Space):
    
    def __init__(self):
        super(VaryingDiscreteSpace, self).__init__((), np.int64)
    
    def sample(self, n):
        return self.np_random.randint(n)
    
    def contains(self, x):
        return x >= 0

class VaryingDiscreteLogSpace(VaryingDiscreteSpace):
    
    def __init__(self):
        super(VaryingDiscreteLogSpace, self).__init__()
    
    def sample(self, n, p=0.99):
        r = self.np_random.logseries(p)
        return n - 1 if r > n else r - 1

class VaryingMemory():
    
    def __init__(self, default = None):
        self.memory = {}
        self.default = default
    
    def set(self, state, action, value):
        if state not in self.memory:
            self.memory[state] = {}
        self.memory[state][action] = value
    
    def get(self, state, action=None):
        if state not in self.memory:
            self.memory[state] = {}
        if action is None:
            return self.memory[state]
        if action not in self.memory[state]:
            self.memory[state][action] = self.default
        return self.memory[state][action]

class MapMatchingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    """
    Initializes MapMatching Environment
    
    Args:
        region (lib.network.Network): the network to map match to
        tracks (lib.tracks.Tracks): the tracks to load
        track_id (dict): dict of identification values of track to process, key = column, value = value
    """
    def __init__(self, network: lib.network.Network, tracks: lib.tracks.Tracks, seed=None):
        self.action_space = VaryingDiscreteSpace()
        self.observation_space = VaryingDiscreteSpace()
        
        self.network = network
        self.tracks = tracks

        self.seed(seed=seed)
    
    def set_track(self, track_selector, buffer=50, subgraph_buffer=1000, prepare=True, prepare_type='triangle', prepare_buffer=(25, 100), state_steps=2, max_skip=3, truncate=False, use_existing_nodes=True, use_new_nodes=True):
        # set track by given id
        self.track_selector = track_selector
        self.track_points, self.track_line = self.tracks.get_track(track_selector, prepare=prepare, prepare_type=prepare_type, prepare_buffer=prepare_buffer)

        # calculate candidates
        if truncate:
            self.graph_clean = self.network.truncate_graph(self.network.graph, self.track_line, buffer=subgraph_buffer, retain_all=False)
        else:
            self.graph_clean = self.network.graph.copy()
        self.nodes_clean, self.edges_clean, self.track_candidates, self.graph_updated, self.nodes_updated, self.edges_updated = self.network.candidate_search(self.graph_clean, points=self.track_points, buffer=buffer, use_existing=use_existing_nodes, use_new=use_new_nodes)
        self.track_list = self.track_line.to_dict(orient='records')
        
        if state_steps < 2:
            raise ValueError("state_steps cannot be smaller than 2 because routes can only be calculated between at least two measurements.")
        if state_steps > len(self.track_points):
            state_steps = len(self.track_points)
        
        self.state_steps = state_steps
        self.max_skip = max_skip
        self.reset_memory()

    def reset_memory(self):
        self.memory = {}
        self.skips = {}
        self.worst_value = float('inf')
        self.routes = {}
        self.route_metrics = {}
        self.route_directions = {}
        self.action_matrix = None
        self.reset_action_selections()

    def reset_action_selections(self):
        self.action_selections = {}

    def estimate_states(self):
        candidate_choices = [len(candidate['candidates']) for candidate in self.track_candidates]
        combinations = 0
        i = 0
        j = 0
        while i < len(candidate_choices) - (self.state_steps - 1):
            combination_part = 1
            for k in range(j - i + 1):
                combination_part *= candidate_choices[i+k]
            combinations += combination_part
            if j - i + 1 >= self.state_steps:
                i += 1
            j += 1
        return combinations

    """
    Returns the action space in the current step.
    
    Returns:
        actions (list): possible actions
    """
    def actions(self):
        # return length of candidates found
        return len(self.track_candidates[self.state[-1]]['candidates']) if self.state[-1] < len(self.track_candidates) else 0

    def _calculate_action_selection(self, curr_state, curr_action, next_state):
        selector = (curr_state, curr_action, next_state)

        candidate_curr = self.track_candidates[curr_state]
        candidate_next = self.track_candidates[next_state]
        candidate_next_length = self.track_list[curr_state]['length']
        candidate_next_bearing = self.track_list[curr_state]['bearing']
        candidate_curr_node = candidate_curr['candidates'][candidate_curr['names'][curr_action]]

        if selector not in self.action_matrix:
            self.action_matrix[selector] = []

        for next_action in range(len(candidate_next['names'])):
            candidate_next_node = candidate_next['candidates'][candidate_next['names'][next_action]]
            candidate_next_line = candidate_next['lines'][candidate_next['names'][next_action]]
            distance = candidate_curr_node['geometry'].distance(candidate_next_node['geometry'])
            bearing = candidate_next_line['bearing']
            angle_diff = abs(self.network.angle_diff(candidate_next_bearing, bearing))
            length_diff = abs(candidate_next_length - distance)
            action_tuple = (angle_diff, length_diff, next_action)

            self.action_matrix[selector].append(action_tuple)

        self.action_matrix[selector].sort(key=lambda x: x[0] + x[1])

    # Return a probable action not taken before
    def sample_intelligent(self, samples=1):
        if self.action_matrix is None:
            self.action_matrix = {}
            for curr_state in range(len(self.track_candidates) - 1):
                for curr_action in range(len(self.track_candidates[curr_state]['names'])):
                    self._calculate_action_selection(curr_state, curr_action, curr_state + 1)

        if len(self.state) >= 3:
            curr_state = self.state[-3]
            curr_action = self.state[-2]

        if len(self.state) >= 1:
            next_state = self.state[-1]
            candidate_next = self.track_candidates[next_state]

            if self.state not in self.action_selections:
                self.action_selections[self.state] = {action: 0 for action in range(len(candidate_next['names']))}

            if len(self.action_selections[self.state]) > 0:
                possible_actions = sorted(self.action_selections[self.state].items(), key=lambda item: item[1])
                possible_action = possible_actions[0][0]  # choose next best action

                if len(self.state) >= 3:
                    selector = (curr_state, curr_action, next_state)
                    if selector not in self.action_matrix:
                        self._calculate_action_selection(curr_state, curr_action, next_state)
                    next_action = self.action_matrix[selector][possible_action][2]
                else:
                    next_action = possible_action

                self.action_selections[self.state][possible_action] += 1
                if self.action_selections[self.state][possible_action] >= samples:
                    del self.action_selections[self.state][possible_action]
            else:
                next_action = -1

        return next_action

    # Return a probable action not taken before
    def sample_greedy(self, samples=1):
        if len(self.state) >= 3:
            curr_state = self.state[-3]
            curr_action = self.state[-2]

        if len(self.state) >= 1:
            next_state = self.state[-1]
            candidate_next = self.track_candidates[next_state]

            if self.state not in self.action_selections:
                self.action_selections[self.state] = {action: 0 for action in range(len(candidate_next['names']))}

            if len(self.action_selections[self.state]) > 0:
                possible_actions = sorted(self.action_selections[self.state].items(), key=lambda item: item[1])
                possible_action = possible_actions[0][0]  # choose next best action

                state = self.state
                action_rewards = {}
                for action, _ in possible_actions:
                    _, reward, done, _ = self.step(action)
                    action_rewards[action] = reward
                    self.state = state

                if len(action_rewards) > 0:
                    possible_action, _ = max(action_rewards.items(), key=lambda x: x[1])

                next_action = possible_action

                self.action_selections[self.state][possible_action] += 1
                if self.action_selections[self.state][possible_action] >= samples:
                    del self.action_selections[self.state][possible_action]
            else:
                next_action = -1

        return next_action

    def step(self, action):
        try:
            # don't use extra variables here as good as possible, the less the more speed
            self.state, reward, done = self.memory[(*self.state, action)]
            return self.state, reward, done, {}
        except KeyError:
            # check if we reached regular finish, may be overwritten later if something went wrong during step
            # state_steps * 2 - 1 = maximum length of state tuple, -2 cuts first state-action pair
            new_state = (*self.state[-(self.state_steps * 2 - 1 - 2):], action, self.state[-1] + 1)
            done = self.state[-1] + 1 >= len(self.track_candidates)

            # initialize all metrics for easy reward computation
            route = None
            node_next_distance = 0
            node_curr_distance = 0
            node_prev_distances = [0]
            candidate_next_length = 0
            candidate_next_bearing = 0
            candidate_next_speed = 0
            candidate_curr_lengths = [0]
            candidate_curr_bearings = [0]
            candidate_curr_speeds = [0]
            candidate_directions = [0]
            route_next_length = 0
            route_next_bearing = 0
            route_next_bearing_mean = 0
            route_next_speed = 0
            route_curr_lengths = [0]
            route_curr_bearings = [0]
            route_curr_bearing_means = [0]
            route_curr_speeds = [0]
            
            # on first edge, reward is only negative distance, no route existing yet
            if len(self.state) >= 1:
                next_state = self.state[-1]
                next_action = action
                
                candidate_next = self.track_candidates[next_state]
                candidate_next_timestamp = candidate_next['point']['timestamp']
                node_next_distance = candidate_next['candidates'][candidate_next['names'][next_action]]['distance']
            
            if len(self.state) >= 3:
                curr_state = self.state[-3]
                curr_action = self.state[-2]
                
                candidate_curr = self.track_candidates[curr_state]
                candidate_curr_timestamp = candidate_curr['point']['timestamp']
                node_curr_distance = candidate_curr['candidates'][candidate_curr['names'][curr_action]]['distance']
            
                # calculate route
                route_next, self.routes = self.network.route(self.graph_updated, candidate_curr, candidate_next, source_select=curr_action, target_select=next_action, routes_cache=self.routes)
                route = route_next['route']
                
                # calculate metrics, route is compared to candidate line
                candidate_next_length = self.track_list[curr_state]['length']
                candidate_next_bearing = self.track_list[curr_state]['bearing']

                route_next_metrics, self.route_metrics = self.network.route_metrics(self.nodes_updated, self.edges_updated, route_next['route'], route_metrics_cache=self.route_metrics)
                route_next_length = route_next_metrics['length']
                route_next_bearing = route_next_metrics['bearing']
                route_next_bearing_mean = route_next_metrics['bearing_mean']
                
                candidate_next_timedelta = candidate_next_timestamp - candidate_curr_timestamp
                candidate_next_speed = candidate_next_length / candidate_next_timedelta.total_seconds() if candidate_next_timedelta.total_seconds() > 0 else 0
                route_next_speed = route_next_length / candidate_next_timedelta.total_seconds() if candidate_next_timedelta.total_seconds() > 0 else 0

            if len(self.state) >= 5:
                prev_states = [self.state[i] for i in range(-len(self.state), -5+1, 2)]
                prev_actions = [self.state[i] for i in range(-len(self.state) + 1, -4+1, 2)]
                
                candidate_prevs = [self.track_candidates[prev_state] for prev_state in prev_states]
                candidate_prevs_timestamps = [candidate_prevs[i]['point']['timestamp'] for i in range(len(prev_states))]
                node_prev_distances = [candidate_prevs[i]['candidates'][candidate_prevs[i]['names'][prev_actions[i]]]['distance'] for i in range(len(prev_states))]
                
                # calculate routes
                route_currs = []
                for i in range(len(prev_states) - 1):
                    route_curr, self.routes = self.network.route(self.graph_updated, candidate_prevs[i], candidate_prevs[i+1], source_select=prev_actions[i], target_select=prev_actions[i+1], routes_cache=self.routes)
                    route_currs.append(route_curr)
                route_curr, self.routes = self.network.route(self.graph_updated, candidate_prevs[-1], candidate_curr, source_select=prev_actions[-1], target_select=curr_action, routes_cache=self.routes)
                route_currs.append(route_curr)
                
                try:
                    route = self.network.routes_join([*route_currs, route_next])
                except ValueError:
                    route = None
                
                # calculate metrics, route is compared to candidate line
                candidate_curr_lengths = [self.track_list[prev_state]['length'] for prev_state in prev_states]
                candidate_curr_bearings = [self.track_list[prev_state]['bearing'] for prev_state in prev_states]
                
                candidate_directions = []
                for i in range(len(prev_states) - 1):
                    candidate_direction = self.tracks.angle_diff(candidate_curr_bearings[i], candidate_curr_bearings[i+1])
                    candidate_directions.append(candidate_direction)
                candidate_direction = self.tracks.angle_diff(candidate_curr_bearings[-1], candidate_next_bearing)
                candidate_directions.append(candidate_direction)
                
                route_curr_lengths = []
                route_curr_bearings = []
                route_curr_bearing_means = []
                for route_curr in route_currs:
                    route_curr_metrics, self.route_metrics = self.network.route_metrics(self.nodes_updated, self.edges_updated, route_curr['route'], route_metrics_cache=self.route_metrics)
                    route_curr_lengths.append(route_curr_metrics['length'])
                    route_curr_bearings.append(route_curr_metrics['bearing'])
                    route_curr_bearing_means.append(route_curr_metrics['bearing_mean'])
                
                candidate_curr_timedeltas = [candidate_prevs_timestamps[i+1] - candidate_prevs_timestamps[i] for i in range(len(prev_states) - 1)]
                candidate_curr_timedeltas.append(candidate_curr_timestamp - candidate_prevs_timestamps[-1])
                candidate_curr_speeds = [candidate_curr_lengths[i] / candidate_curr_timedeltas[i].total_seconds() if candidate_curr_timedeltas[i].total_seconds() > 0 else 0 for i in range(len(route_curr_lengths))]
                route_curr_speeds = [route_curr_lengths[i] / candidate_curr_timedeltas[i].total_seconds() if candidate_curr_timedeltas[i].total_seconds() > 0 else 0 for i in range(len(route_curr_lengths))]

            route_direction = 0
            route_error = 0
            if route is not None and len(route) >= 3:
                route_direction = 0
                for i in range(1, len(route) - 1):
                    route_single_direction, self.route_directions = self.network.calculate_direction(self.edges_updated, u=route[i - 1], v=route[i], w=route[i + 1], directions_cache=self.route_directions)
                    route_direction += route_single_direction
            
            reward = -(1 * abs(sum(node_prev_distances) + node_curr_distance + node_next_distance) +
                       1 * abs(sum(candidate_curr_lengths) - sum(route_curr_lengths)) +
                       1 * abs(sum([candidate_curr_speeds[i] - route_curr_speeds[i] for i in range(len(candidate_curr_speeds))])) +
                       1 * abs(candidate_next_length - route_next_length) +
                       1 * abs(candidate_next_speed - route_next_speed) +
                       1 * abs(sum([self.tracks.angle_diff(candidate_curr_bearings[i], route_curr_bearings[i]) for i in range(len(candidate_curr_bearings))])) +
                       1 * abs(self.tracks.angle_diff(candidate_next_bearing, route_next_bearing)) +
                       1 * abs(self.tracks.angle_diff(sum(candidate_directions), route_direction)) +
                       route_error)
            
            if reward < self.worst_value:
                self.worst_value = reward
            
            skip = None
            if len(self.state) >= 3:
                # add error if no route is found but we searched
                if route is None:
                    reward += -abs(self.worst_value) * 2 # double worst value
                    skip = 'no_route'
                    done = True

                # add error if speed is faster than 180 km/h
                if route is not None and (route_next_length != 0 and route_next_speed == 0 or route_next_speed > 50):
                    reward += -abs(self.worst_value) * 2 # double worst value
                    #skip = 'fast_travel'
                
                # add error if route is 10 times longer than distance
                if route is not None and (route_next_length != 0 and candidate_next_length != 0 and route_next_length / candidate_next_length > 10):
                    reward += -abs(self.worst_value) * 2 # double worst value
                    #skip = 'long_route'
            
            # State-Action-State-Action-State-Action defines partial route
            selector = (*self.state, action)
            
            # skip this connection, skip is None if len(self.state) < 3, only skip up to max_skip times, else it might skip until end
            if skip is not None and next_state - curr_state <= self.max_skip:
                new_state = (*self.state[:-1], self.state[-1] + 1)
                done = new_state[-1] + 1 >= len(self.track_candidates)
                self.skips[selector] = (new_state, done, skip)

            self.memory[selector] = (new_state, reward, done)
            self.state = new_state

            return self.state, reward, done, {}

    def reset(self):
        # (s0, a0, s1, a1, s2)
        self.state = (0,)
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

class MapMatchingLearning():

    def _qlearning(self, env, learning_rate=0.9, learning_decay=None, learning_decay_linear=False, learning_min=0.01, discount=0.99, epsilon=0.1, epsilon_decay=None, epsilon_decay_linear=False, epsilon_min=0.01, episodes=100, improvement_break=10, prints=False):
        scores = []
        bests = []
        durations = []
        e = epsilon
        a = learning_rate
        current_best = float('-inf')
        current_best_counter = 0
    
        # Initializing the Q-table of size state-space x action-space with zeros
        Q = VaryingMemory(default = 0)
    
        for episode in range(episodes):
            start = time.time()
            # Reset the game-state, done and score before every episode
            state = env.reset()
            done = False
            score = 0
    
            while not done:
                # With the probabilty of (1 - epsilon) take the best action in our Q-table
                if env.np_random.uniform(0, 1) > e:
                    action = max(Q.get(state), key=Q.get(state).get) if Q.get(state) else 0
                # Else take a random action
                else:
                    action = env.action_space.sample(env.actions())

                # Step the game forward
                next_state, reward, done, _ = env.step(action)
                #print(state, action, reward, next_state)
    
                # Add up the score
                score += reward
    
                # Update our Q-table with our Q-function
                Q.set(state, action, Q.get(state, action) + a * (reward + discount * (max(Q.get(next_state).values()) if Q.get(next_state) else 0) - Q.get(state, action)))
    
                # Set the next state as the current state
                state = next_state
    
            scores.append(score)
            
            # get current policy
            policy = collections.OrderedDict()
            state = env.reset()
            done = False
            best = 0
            while not done:
                action = max(Q.get(state), key=Q.get(state).get) if Q.get(state) else 0
                policy[state[-1]] = action
                
                state, reward, done, _ = env.step(action)
                best += reward
            
            bests.append(best)

            if not math.isclose(best, current_best, rel_tol=0.001):
                current_best = best
                current_best_counter = 0
            else:
                current_best_counter += 1

            # Reducing our learning rate each episode (Baking learning results)
            if learning_decay is not None and a >= learning_min:
                a = a - learning_decay if learning_decay_linear else a * learning_decay
    
            # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
            if epsilon_decay is not None and e >= epsilon_min:
                e = e - epsilon_decay if epsilon_decay_linear else e * epsilon_decay

            end = time.time()
            duration = end - start
            durations.append(duration)

            if prints:
                print("Episode: {}/{}, score: {:.4f}, duration: {:.4f}, best: {:.4f}, epsilon: {:.4f}, learning_rate: {:.4f}".format(episode+1, episodes, score, duration, best, e, a))

            if current_best_counter >= improvement_break:
                if prints:
                    print("Breaking at episode {}/{} because we had optimal actions for {} succeeding episodes.".format(episode + 1, episodes, improvement_break))
                break

        return {'policy': policy, 'scores': scores, 'bests': bests, 'durations': durations, 'Q': Q}

    def _double_qlearning(self, env, learning_rate=0.9, learning_decay=None, learning_decay_linear=False, learning_min=0.01, discount=0.99, epsilon=0.1, epsilon_decay=None, epsilon_decay_linear=False, epsilon_min=0.01, episodes=100, improvement_break=10, prints=False):
        scores = []
        bests = []
        durations = []
        e = epsilon
        a = learning_rate
        current_best = float('-inf')
        current_best_counter = 0

        # Initializing the Q-table of size state-space x action-space with zeros
        Qa = VaryingMemory(default = 0)
        Qb = VaryingMemory(default = 0)
    
        for episode in range(episodes):
            start = time.time()
            # Reset the game-state, done and score before every episode
            state = env.reset()
            done = False
            score = 0
    
            while not done:
                # With the probabilty of (1 - epsilon) take the best action in our Q-table
                if env.np_random.uniform(0, 1) > e:
                    Qsum = Qa.get(state).copy()
                    for i in Qb.get(state).keys():
                        if i not in Qsum:
                            Qsum[i] = 0
                        Qsum[i] += Qb.get(state, i)

                    action = max(Qsum, key=Qsum.get) if Qsum else 0
                # Else take a random action
                else:
                    action = env.action_space.sample(env.actions())

                # Step the game forward
                next_state, reward, done, _ = env.step(action)
                #print(state, action, reward, next_state)
    
                # Add up the score
                score += reward
    
                # Update our Q-table with our Q-function
                if env.np_random.uniform(0, 1) > 0.5:
                    Qa.set(state, action, Qa.get(state, action) + a * (reward + discount * (Qb.get(next_state, max(Qa.get(next_state), key=Qa.get(next_state).get)) if Qa.get(next_state) else 0) - Qa.get(state, action)))
                else:
                    Qb.set(state, action, Qb.get(state, action) + a * (reward + discount * (Qa.get(next_state, max(Qb.get(next_state), key=Qb.get(next_state).get)) if Qb.get(next_state) else 0) - Qb.get(state, action)))
    
                # Set the next state as the current state
                state = next_state
    
            scores.append(score)
            
            # get current policy
            policy = collections.OrderedDict()
            state = env.reset()
            done = False
            best = 0
            while not done:
                Qsum = Qa.get(state).copy()
                for i in Qb.get(state).keys():
                    if i not in Qsum:
                        Qsum[i] = 0
                    Qsum[i] += Qb.get(state, i)
                
                action = max(Qsum, key=Qsum.get) if len(Qsum) > 0 else 0
                policy[state[-1]] = action
                
                state, reward, done, _ = env.step(action)
                best += reward
            
            bests.append(best)

            if not math.isclose(best, current_best, rel_tol=0.001):
                current_best = best
                current_best_counter = 0
            else:
                current_best_counter += 1

            # Reducing our learning rate each episode (Baking learning results)
            if learning_decay is not None and a >= learning_min:
                a = a - learning_decay if learning_decay_linear else a * learning_decay
    
            # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
            if epsilon_decay is not None and e >= epsilon_min:
                e = e - epsilon_decay if epsilon_decay_linear else e * epsilon_decay

            end = time.time()
            duration = end - start
            durations.append(duration)

            if prints:
                print("Episode: {}/{}, score: {:.4f}, duration: {:.4f}, best: {:.4f}, epsilon: {:.4f}, learning_rate: {:.4f}".format(episode+1, episodes, score, duration, best, e, a))

            if current_best_counter >= improvement_break:
                if prints:
                    print("Breaking at episode {}/{} because we had optimal actions for {} succeeding episodes.".format(episode + 1, episodes, improvement_break))
                break

        return {'policy': policy, 'scores': scores, 'bests': bests, 'durations': durations, 'Qa': Qa, 'Qb': Qb}

    def _sarsa(self, env, learning_rate=0.9, learning_decay=None, learning_decay_linear=False, learning_min=0.01, discount=0.99, epsilon=0.1, epsilon_decay=None, epsilon_decay_linear=False, epsilon_min=0.01, episodes=100, improvement_break=10, prints=False):
        scores = []
        bests = []
        durations = []
        e = epsilon
        a = learning_rate
        current_best = float('-inf')
        current_best_counter = 0

        # Initializing the Q-table of size state-space x action-space with zeros
        Q = VaryingMemory(default = 0)
    
        for episode in range(episodes):
            start = time.time()
            # Reset the game-state, done and score before every episode
            state = env.reset()
            done = False
            score = 0
    
            # With the probabilty of (1 - epsilon) take the best action in our Q-table
            if env.np_random.uniform(0, 1) > e:
                action = max(Q.get(state), key=Q.get(state).get) if Q.get(state) else 0
            # Else take a random action
            else:
                action = env.action_space.sample(env.actions())
    
            while not done:
                # Step the game forward
                next_state, reward, done, _ = env.step(action)
                #print(state, action, reward, next_state)
    
                if not done:
                    # With the probabilty of (1 - epsilon) take the best action in our Q-table
                    if env.np_random.uniform(0, 1) > e:
                        next_action = max(Q.get(next_state), key=Q.get(next_state).get) if Q.get(next_state) else 0
                    # Else take a random action
                    else:
                        next_action = env.action_space.sample(env.actions())
    
                # Add up the score
                score += reward
    
                # Update our Q-table with our Q-function
                Q.set(state, action, Q.get(state, action) + a * (reward + discount * (Q.get(next_state, next_action) if Q.get(next_state) else 0) - Q.get(state, action)))
    
                # Set the next state as the current state
                state = next_state
                
                action = next_action
    
            scores.append(score)
            
            # get current policy
            policy = collections.OrderedDict()
            state = env.reset()
            done = False
            best = 0
            while not done:
                action = max(Q.get(state), key=Q.get(state).get) if Q.get(state) else 0
                policy[state[-1]] = action
                
                state, reward, done, _ = env.step(action)
                best += reward
            
            bests.append(best)

            if not math.isclose(best, current_best, rel_tol=0.001):
                current_best = best
                current_best_counter = 0
            else:
                current_best_counter += 1

            # Reducing our learning rate each episode (Baking learning results)
            if learning_decay is not None and a >= learning_min:
                a = a - learning_decay if learning_decay_linear else a * learning_decay
    
            # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
            if epsilon_decay is not None and e >= epsilon_min:
                e = e - epsilon_decay if epsilon_decay_linear else e * epsilon_decay

            end = time.time()
            duration = end - start
            durations.append(duration)

            if prints:
                print("Episode: {}/{}, score: {:.4f}, duration: {:.4f}, best: {:.4f}, epsilon: {:.4f}, learning_rate: {:.4f}".format(episode+1, episodes, score, duration, best, e, a))

            if current_best_counter >= improvement_break:
                if prints:
                    print("Breaking at episode {}/{} because we had optimal actions for {} succeeding episodes.".format(episode + 1, episodes, improvement_break))
                break

        return {'policy': policy, 'scores': scores, 'bests': bests, 'durations': durations, 'Q': Q}

    def _expected_sarsa(self, env, learning_rate=0.9, learning_decay=None, learning_decay_linear=False, learning_min=0.01, discount=0.99, epsilon=0.1, epsilon_decay=None, epsilon_decay_linear=False, epsilon_min=0.01, episodes=100, improvement_break=10, prints=False):
        scores = []
        bests = []
        durations = []
        e = epsilon
        a = learning_rate
        current_best = float('-inf')
        current_best_counter = 0

        # Initializing the Q-table of size state-space x action-space with zeros
        Q = VaryingMemory(default=0)

        for episode in range(episodes):
            start = time.time()
            # Reset the game-state, done and score before every episode
            state = env.reset()
            done = False
            score = 0

            while not done:
                # With the probabilty of (1 - epsilon) take the best action in our Q-table
                if env.np_random.uniform(0, 1) > e:
                    action = max(Q.get(state), key=Q.get(state).get) if Q.get(state) else 0
                # Else take a random action
                else:
                    action = env.action_space.sample(env.actions())

                # Step the game forward
                next_state, reward, done, _ = env.step(action)
                # print(state, action, reward, next_state)

                # Add up the score
                score += reward

                # Update our Q-table with our Expected SARSA function
                best_action = max(Q.get(next_state), key=Q.get(next_state).get) if Q.get(next_state) else 0
                expectation = (1 - epsilon) * Q.get(next_state, best_action) if Q.get(next_state) else 0 +\
                              (epsilon / env.actions() if env.actions() > 0 else 0) * sum(Q.get(next_state, next_action) for next_action in range(env.actions()))
                Q.set(state, action, Q.get(state, action) + a * (reward + discount * expectation - Q.get(state, action)))

                # Set the next state as the current state
                state = next_state

            scores.append(score)

            # get current policy
            policy = collections.OrderedDict()
            state = env.reset()
            done = False
            best = 0
            while not done:
                action = max(Q.get(state), key=Q.get(state).get) if Q.get(state) else 0
                policy[state[-1]] = action

                state, reward, done, _ = env.step(action)
                best += reward

            bests.append(best)

            if not math.isclose(best, current_best, rel_tol=0.001):
                current_best = best
                current_best_counter = 0
            else:
                current_best_counter += 1

            # Reducing our learning rate each episode (Baking learning results)
            if learning_decay is not None and a >= learning_min:
                a = a - learning_decay if learning_decay_linear else a * learning_decay

            # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
            if epsilon_decay is not None and e >= epsilon_min:
                e = e - epsilon_decay if epsilon_decay_linear else e * epsilon_decay

            end = time.time()
            duration = end - start
            durations.append(duration)

            if prints:
                print("Episode: {}/{}, score: {:.4f}, duration: {:.4f}, best: {:.4f}, epsilon: {:.4f}, learning_rate: {:.4f}".format(episode + 1, episodes, score, duration, best, e, a))

            if current_best_counter >= improvement_break:
                if prints:
                    print("Breaking at episode {}/{} because we had optimal actions for {} succeeding episodes.".format(episode + 1, episodes, improvement_break))
                break

        return {'policy': policy, 'scores': scores, 'bests': bests, 'durations': durations, 'Q': Q}

    def qlearning(self, env, learning_rate=1.0, discount=1, epsilon=0.5, epsilon_raise=0.9, epsilon_reduce=0.99, epsilon_max=1.0, epsilon_min=0.1, episodes=100, improvement_break=10, improvment_reset=None, resets=3, samples=3, intelligent=True, prints=False):
        scores = []
        bests = []
        durations = []
        e = epsilon
        a = learning_rate
        max_action_counter = 0
        current_best = float('-inf')
        current_best_counter = 0
        reset_counter = 0

        if improvment_reset is None:
            improvment_reset = improvement_break / 2

        env.reset_action_selections()

        # Initializing the Q-table of size state-space x action-space with zeros
        Q = VaryingMemory(default=0)

        for episode in range(episodes):
            start = time.time()

            # Reset the game-state, done and score before every episode
            state = env.reset()
            done = False
            score = 0
            max_action = -1

            while not done:
                # With the probability of epsilon take a random action
                if env.np_random.uniform(0, 1) <= e:
                    if intelligent:
                        action = env.sample_intelligent(samples)
                    else:
                        action = env.sample_greedy(samples)
                # Else take the best action in our Q-table, mark this with -1
                else:
                    action = -1

                if action > max_action:
                    max_action = action

                if action == -1:
                    action = max(Q.get(state), key=Q.get(state).get) if Q.get(state) else 0

                # Step the game forward
                next_state, reward, done, _ = env.step(action)
                # print(state, action, reward, next_state)

                # Add up the score
                score += reward

                # Update our Q-table with our Q-function
                Q.set(state, action, Q.get(state, action) +
                      a * (reward + discount * (max(Q.get(next_state).values()) if Q.get(next_state) else 0) -
                           Q.get(state, action)))

                # Set the next state as the current state
                state = next_state

            scores.append(score)

            # get current policy
            policy = collections.OrderedDict()
            state = env.reset()
            done = False
            best = 0
            while not done:
                action = max(Q.get(state), key=Q.get(state).get) if Q.get(state) else 0
                policy[state[-1]] = action

                state, reward, done, _ = env.step(action)
                best += reward

            bests.append(best)

            if max_action == -1:
                max_action_counter += 1
            else:
                max_action_counter = 0

            if not math.isclose(best, current_best, rel_tol=0.001):
                current_best = best
                current_best_counter = 0
            else:
                current_best_counter += 1

            # Varying our epsilon, raise towards 1.0 when we have not found better actions, else reduce
            if epsilon_min <= e <= epsilon_max:
                if max_action_counter == 0 and epsilon_reduce is not None and epsilon_reduce > 0.0:
                    e *= epsilon_reduce
                elif max_action_counter > 0 and epsilon_raise is not None and epsilon_raise > 0.0:
                    e /= epsilon_raise
                e = min(epsilon_max, max(epsilon_min, e))

            if reset_counter < resets and current_best_counter >= improvment_reset:
                env.reset_action_selections()
                reset_counter += 1
                current_best_counter = 0

            end = time.time()
            duration = end - start
            durations.append(duration)

            if prints:
                print("Episode: {}/{}, score: {:.4f}, duration: {:.4f}, best: {:.4f}, epsilon: {:.4f}, learning_rate: {:.4f}".format(
                    episode + 1, episodes, score, duration, best, e, a))

            if current_best_counter >= improvement_break:
                if prints:
                    print("Breaking at episode {}/{} because we had optimal actions for {} succeeding episodes.".format(episode + 1, episodes, improvement_break))
                break

        env.reset_action_selections()

        return {'policy': policy, 'scores': scores, 'bests': bests, 'durations': durations, 'Q': Q}

    def generate_states(self, env):
        state = env.reset()
        states_to_visit = {state}
        states = {state}

        while states_to_visit:
            state = states_to_visit.pop()
            env.state = state
            for action in range(env.actions()):
                new_state, reward, done, _ = env.step(action)
                if not done and new_state not in states:
                    states_to_visit.add(new_state)
                states.add(new_state)
                env.state = state

        return states
    
    def value_iteration(self, env, threshold=0.001, discount=1.0, prints=False):
        if prints:
            print("Generating state values...", end=" ")
            
        states = self.generate_states(env)
        states = sorted(states)
        V = {state: 0 for state in states}
        
        if prints:
            print("done")
        
        values = []
        delta = threshold * 2
        episode = 0
        while delta > threshold and episode <= len(states):
            old_value = sum(V.values())

            for state in states:
                env.state = state
                action_values = []
                for action in range(env.actions()):
                    new_state, reward, done, _ = env.step(action)
                    if new_state in V: # only if new_state is valid
                        state_value = reward + discount * V[new_state]
                        action_values.append(state_value)
                    env.state = state
                V[state] = max(action_values) if len(action_values) > 0 else 0
            
            value = sum(V.values())
            
            new_delta = abs(value - old_value)
            if new_delta < delta:
                delta = new_delta
                
            values.append(value)

            if prints:
                print("Episode: {}, value: {}".format(episode+1, value))
            
            episode += 1
        
        if prints:
            print("Evaluating policy...", end=" ")
            
        state = env.reset()
        done = False
        score = 0
        actions = collections.OrderedDict()

        while not done:
            action_values = []
            for action in range(env.actions()):
                new_state, reward, done, _ = env.step(action)
                if new_state in V:
                    state_value = reward + discount * V[new_state]
                    action_values.append(state_value)
                env.state = state
            
            if len(action_values) == 0:
                break # done, no more actions found
                
            action, _ = max(enumerate(action_values), key=lambda x: x[1])
            actions[state[-1]] = action
            
            state, reward, done, _ = env.step(action)
            
            score += reward
        
        policy = actions
        
        if prints:
            print("done, score: {}".format(score))
        
        return {'policy': policy, 'values': values, 'best': score}

    def random(self, env, episodes=100, prints=False):
        # Initializing the list of scores
        scores = []

        for episode in range(episodes):
            # Reset the state, done and score before every episode
            env.reset()
            done = False
            score = 0
            actions = collections.OrderedDict()
    
            while not done:
                # Act randomly until done or maximum steps reached
                state = env.state
                action = env.action_space.sample(env.actions())
                actions[state[-1]] == action
                
                _, reward, done, _ = env.step(action)
                
                score += reward

            scores.append(score)
            
            if score >= max(scores):
                policy = actions
                best = score
            
            if prints:
                print("Episode: {}/{}, score: {}".format(episode+1, episodes, score))

        return {'policy': policy, 'scores': scores, 'best': best}
    
    def greedy(self, env, prints=False):
        # Initializing the list of scores
        scores = []

        # Reset the state, done and score before every episode
        state = env.reset()
        done = False
        score = 0
        actions = collections.OrderedDict()

        while not done:
            # Act greedy
            action_rewards = []
            for action in range(env.actions()):
                _, reward, done, _ = env.step(action)
                action_rewards.append(reward)
                env.state = state
            
            if len(action_rewards) == 0:
                break # done, no more actions found
                
            action, _ = max(enumerate(action_rewards), key=lambda x: x[1])
            actions[state[-1]] = action
            
            state, reward, done, _ = env.step(action)
            
            score += reward

        scores.append(score)
        
        policy = actions
        
        if prints:
            print("Score: {}".format(score))

        return {'policy': policy, 'scores': scores, 'best': score}
    
    def nearest(self, env, prints=False):
        # Initializing the list of scores
        scores = []

        # Reset the state, done and score before every episode
        env.reset()
        done = False
        score = 0
        actions = collections.OrderedDict()

        while not done:
            # only take nearest action
            state = env.state
            action = 0
            actions[state[-1]] = action
            
            _, reward, done, _ = env.step(action)
            
            score += reward

        scores.append(score)
        
        policy = actions
        
        if prints:
            print("Score: {}".format(score))

        return {'policy': policy, 'scores': scores, 'best': score}

    def export_agent(self, agent, title="Learning Agent", plot=True, image_file="images/learning.png", image_format="png", dpi=300, show=True, export=True, export_file="exports/learning.csv"):
        scores = agent['values'] if 'values' in agent else agent['scores']

        if plot:
            best = max(agent['bests']) if 'bests' in agent else agent['best']
            
            fig, ax = plt.subplots()
            plt.title(title)
            marker, = ax.plot(scores, "k-" if 'values' in agent else "ko", markersize=0.3, linewidth=0.3)
            if 'bests' in agent:
                marker, = ax.plot(agent['bests'], "b-", linewidth=0.2)
            # if 'durations' in agent:
            #     durations = agent['durations']
            #     rescale = (min(scores), max(scores))
            #     scale = (min(durations), max(durations))
            #     durations_rescaled = list(map(lambda x: (x / scale[1] - scale[0]) * rescale[0], durations))
            #     marker, = ax.plot(durations_rescaled, "-m", linewidth=0.2)
            plt.legend([marker], ["Best: {:.2f}".format(best)], markerscale=10)
            plt.xlabel('Episodes')
            plt.ylabel('Score' if 'scores' in agent else 'Value')
            plt.savefig(image_file, dpi=dpi, format=image_format)
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        if export:
            df = pd.DataFrame({'values' if 'values' in agent else 'scores': scores})
            if 'bests' in agent:
                df['bests'] = agent['bests']
            df.to_csv(export_file)
