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

import os

import pandas as pd


def export_column(dataframe, column, filepattern, sort=False, reverse=False, percentile=False):
    if type(dataframe) == pd.Series:
        tmp = dataframe
    else:
        tmp = dataframe[column]
    if sort:
        tmp = tmp.sort_values(ascending=not reverse).reset_index()
    tmp['index'] = tmp.index
    if percentile:
        tmp['index'] = tmp['index'] / tmp['index'].max()
    tmp = tmp.rename(columns={'index': 'x', column: 'y'})
    tmp.to_csv(filepattern.format(column), index=False)


export_pattern = os.path.join('benchmarks', 'benchmark_{}.csv')
df = pd.read_csv("benchmark_all.csv")

count = df.count()[0]


def error_row(name, column):
    print(name,
          "&", "${:.2f} \\%$".format((df[df[column] <= 0.0].count()[0] / count * 100)),
          "&", "${:.2f}$".format(df[df[column] < float('inf')][column].mean()),
          "&", "${:.2f}$".format(df[df[column] < float('inf')][column].quantile(0.90)),
          "&", "${:.2f}$".format(df[df[column] < float('inf')][column].quantile(0.95)),
          "&", "${:.2f}$".format(df[df[column] < float('inf')][column].quantile(0.99)),
          "\\\\")


def duration_row(name, base, column):
    tmp = df[base] / df[column]
    print(name,
          "&", "${:.2f} \\%$".format((df[tmp > 1.0].count()[0] / count * 100)),
          "&", "${:.2f}$".format(tmp.quantile(0.01)),
          "&", "${:.2f}$".format(tmp.quantile(0.25)),
          "&", "${:.2f}$".format(tmp.median()),
          "&", "${:.2f}$".format(tmp.quantile(0.75)),
          "&", "${:.2f}$".format(tmp.quantile(0.99)),
          "&", "${:.2f}$".format(tmp.mean()),
          "\\\\")


def duration_state_space(name, column):
    print(name,
          "&", "${:.2f}$".format(df[column].quantile(0.01)),
          "&", "${:.2f}$".format(df[column].quantile(0.10)),
          "&", "${:.2f}$".format(df[column].quantile(0.25)),
          "&", "${:.2f}$".format(df[column].median()),
          "&", "${:.2f}$".format(df[column].quantile(0.75)),
          "&", "${:.2f}$".format(df[column].quantile(0.90)),
          "&", "${:.2f}$".format(df[column].quantile(0.99)),
          "&", "${:.2f}$".format(df[column].mean()),
          "\\\\")


print("\nError fraction table\n")
error_row("Value Iteration", 'value_iteration_error_fraction')
error_row("Q-Learning $\\epsilon$-greedy", 'qlearning_epsilon_error_fraction')
error_row("Q-Learning $\\epsilon$-decay", 'qlearning_epsilon_decay_error_fraction')
error_row("Q-Learning* Intelligent", 'qlearning_intelligent_error_fraction')
error_row("Q-Learning* Greedy", 'qlearning_greedy_error_fraction')
error_row("Expected SARSA", 'expected_sarsa_error_fraction')
error_row("Greedy", 'greedy_error_fraction')
error_row("Nearest", 'nearest_error_fraction')

print("\nDuration table\n")
duration_row("Value Iteration", 'value_iteration_duration', 'value_iteration_duration')
duration_row("Q $\\epsilon$-greedy", 'value_iteration_duration', 'qlearning_epsilon_duration')
duration_row("Q $\\epsilon$-decay", 'value_iteration_duration', 'qlearning_epsilon_decay_duration')
duration_row("Q* Intelligent", 'value_iteration_duration', 'qlearning_intelligent_duration')
duration_row("Q* Greedy", 'value_iteration_duration', 'qlearning_greedy_duration')
duration_row("Expected SARSA", 'value_iteration_duration', 'expected_sarsa_duration')
duration_row("Greedy", 'value_iteration_duration', 'greedy_duration')
duration_row("Nearest", 'value_iteration_duration', 'nearest_duration')

print("\nState space exploration table\n")
duration_state_space("Value Iteration", 'value_iteration_states_percentage')
duration_state_space("Q-Learning $\\epsilon$-greedy", 'qlearning_epsilon_states_percentage')
duration_state_space("Q-Learning $\\epsilon$-decay", 'qlearning_epsilon_decay_states_percentage')
duration_state_space("Q-Learning* Intelligent", 'qlearning_intelligent_states_percentage')
duration_state_space("Q-Learning* Greedy", 'qlearning_greedy_states_percentage')
duration_state_space("Expected SARSA", 'expected_sarsa_states_percentage')
duration_state_space("Greedy", 'greedy_states_percentage')
duration_state_space("Nearest", 'nearest_states_percentage')

# error fraction
export_column(df, 'value_iteration_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'nearest_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'greedy_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'qlearning_intelligent_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'qlearning_greedy_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'qlearning_epsilon_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'qlearning_epsilon_decay_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'expected_sarsa_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'expected_sarsa_decay_error_fraction', export_pattern, sort=True, reverse=True, percentile=True)

# duration
export_column(df['value_iteration_duration'] / df['value_iteration_duration'], 'value_iteration_duration', export_pattern, sort=True, reverse=False, percentile=True)
export_column(df['value_iteration_duration'] / df['nearest_duration'], 'nearest_duration', export_pattern, sort=True, reverse=False, percentile=True)
export_column(df['value_iteration_duration'] / df['greedy_duration'], 'greedy_duration', export_pattern, sort=True, reverse=False, percentile=True)
export_column(df['value_iteration_duration'] / df['qlearning_intelligent_duration'], 'qlearning_intelligent_duration', export_pattern, sort=True, reverse=False, percentile=True)
export_column(df['value_iteration_duration'] / df['qlearning_greedy_duration'], 'qlearning_greedy_duration', export_pattern, sort=True, reverse=False, percentile=True)
export_column(df['value_iteration_duration'] / df['qlearning_epsilon_duration'], 'qlearning_epsilon_duration', export_pattern, sort=True, reverse=False, percentile=True)
export_column(df['value_iteration_duration'] / df['qlearning_epsilon_decay_duration'], 'qlearning_epsilon_decay_duration', export_pattern, sort=True, reverse=False, percentile=True)
export_column(df['value_iteration_duration'] / df['expected_sarsa_duration'], 'expected_sarsa_duration', export_pattern, sort=True, reverse=False, percentile=True)
export_column(df['value_iteration_duration'] / df['expected_sarsa_decay_duration'], 'expected_sarsa_decay_duration', export_pattern, sort=True, reverse=False, percentile=True)

# state space
export_column(df, 'value_iteration_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'nearest_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'greedy_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'qlearning_intelligent_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'qlearning_greedy_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'qlearning_epsilon_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'qlearning_epsilon_decay_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'expected_sarsa_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
export_column(df, 'expected_sarsa_decay_states_percentage', export_pattern, sort=True, reverse=True, percentile=True)
