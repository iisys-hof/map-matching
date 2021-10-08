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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

df = pd.read_csv("benchmark_all.csv")
df = df.dropna()
df = df[df['value_iteration_error_fraction'] < float('inf')]
df = df[df['qlearning_intelligent_error_fraction'] < float('inf')]
df = df[df['qlearning_greedy_error_fraction'] < float('inf')]
df = df[df['qlearning_epsilon_error_fraction'] < float('inf')]
df = df[df['qlearning_epsilon_decay_error_fraction'] < float('inf')]
df = df[df['expected_sarsa_error_fraction'] < float('inf')]
df = df[df['expected_sarsa_decay_error_fraction'] < float('inf')]
df = df[df['greedy_error_fraction'] < float('inf')]
df = df[df['nearest_error_fraction'] < float('inf')]

count = df.count()[0]

# error fraction comparison
fig, ax = plt.subplots()
ax.xaxis.grid()
ax.yaxis.grid()
ax.xaxis.set_major_formatter(mtick.PercentFormatter(count))
# plt.xlim(-10, 1 * count)
plt.ylim(-0.1, 1)
# plt.xscale('log')
# plt.yscale('symlog')
ax.plot(df['value_iteration_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_intelligent_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_greedy_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_epsilon_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_epsilon_decay_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['expected_sarsa_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['expected_sarsa_decay_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['greedy_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['nearest_error_fraction'].sort_values(ascending=False).reset_index(drop=True))
ax.legend(['value_iteration', 'qlearning_intelligent', 'qlearning_greedy', 'qlearning_epsilon', 'qlearning_epsilon_decay', 'expected_sarsa', 'expected_sarsa_decay', 'greedy', 'nearest'], loc='upper right')
plt.ylabel('error fraction')
plt.xlabel('percentile of results')
plt.savefig('stats_error_fraction.png', dpi=300)
plt.show()  
plt.close(fig)

# duration comparison
fig, ax = plt.subplots()
ax.xaxis.grid()
ax.yaxis.grid()
ax.xaxis.set_major_formatter(mtick.PercentFormatter(count))
# plt.xlim(-10, 1 * count)
plt.ylim(-0.1, 10)
# plt.xscale('log')
# plt.yscale('symlog')
ax.plot(df['value_iteration_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_intelligent_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_greedy_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_epsilon_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_epsilon_decay_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['expected_sarsa_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['expected_sarsa_decay_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['greedy_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['nearest_duration'].sort_values(ascending=False).reset_index(drop=True))
ax.legend(['value_iteration', 'qlearning_intelligent', 'qlearning_greedy', 'qlearning_epsilon', 'qlearning_epsilon_decay', 'expected_sarsa', 'expected_sarsa_decay', 'greedy', 'nearest'], loc='upper right')
plt.ylabel('duration in seconds')
plt.xlabel('percentile of results')
plt.savefig('stats_duration.png', dpi=300)
plt.show()
plt.close(fig)

# states comparison
fig, ax = plt.subplots()
ax.xaxis.grid()
ax.yaxis.grid()
ax.xaxis.set_major_formatter(mtick.PercentFormatter(count))
# plt.xlim(-10, 1 * count)
plt.ylim(-0.01, 1.01)
# plt.xscale('log')
# plt.yscale('symlog')
ax.plot(df['value_iteration_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_intelligent_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_greedy_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_epsilon_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['qlearning_epsilon_decay_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['expected_sarsa_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['expected_sarsa_decay_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['greedy_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.plot(df['nearest_states_percentage'].sort_values(ascending=False).reset_index(drop=True))
ax.legend(['value_iteration', 'qlearning_intelligent', 'qlearning_greedy', 'qlearning_epsilon', 'qlearning_epsilon_decay', 'expected_sarsa', 'expected_sarsa_decay', 'greedy', 'nearest'], loc='upper right')
plt.ylabel('states percentage')
plt.xlabel('percentile of results')
plt.savefig('stats_states_percentage.png', dpi=300)
plt.show()
plt.close(fig)

# states and error regression comparison
fig, ax = plt.subplots()
ax.xaxis.grid()
ax.yaxis.grid()
plt.xlim(-0.05, 1.05)
plt.ylim(-0.5, 10.5)
ax.scatter(df['qlearning_intelligent_states_percentage'], df['qlearning_intelligent_error_fraction'], zorder=5)
ax.scatter(df['qlearning_greedy_states_percentage'], df['qlearning_greedy_error_fraction'], zorder=5)
ax.scatter(df['qlearning_epsilon_states_percentage'], df['qlearning_epsilon_error_fraction'], zorder=4)
ax.scatter(df['qlearning_epsilon_decay_states_percentage'], df['qlearning_epsilon_decay_error_fraction'], zorder=3)
ax.scatter(df['expected_sarsa_states_percentage'], df['expected_sarsa_error_fraction'], zorder=4)
ax.scatter(df['expected_sarsa_decay_states_percentage'], df['expected_sarsa_decay_error_fraction'], zorder=3)
ax.scatter(df['greedy_states_percentage'], df['greedy_error_fraction'], zorder=2)
ax.scatter(df['nearest_states_percentage'], df['nearest_error_fraction'], zorder=1)
ax.legend(['qlearning_intelligent', 'qlearning_greedy', 'qlearning_epsilon', 'qlearning_epsilon_decay', 'expected_sarsa', 'expected_sarsa_decay', 'greedy', 'nearest'], loc='upper right')
plt.ylabel('error fraction')
plt.xlabel('states percentage')
plt.savefig('stats_states_error_regression.png', dpi=300)
plt.show()
plt.close(fig)

# comparison of error derivation
fig, ax = plt.subplots()
ax.xaxis.grid()
ax.yaxis.grid()
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.5, 3)
ax.boxplot([df['qlearning_intelligent_error_fraction'],
            df['qlearning_greedy_error_fraction'],
            df['qlearning_epsilon_error_fraction'],
            df['qlearning_epsilon_decay_error_fraction'],
            df['expected_sarsa_error_fraction'],
            df['expected_sarsa_decay_error_fraction'],
            df['greedy_error_fraction'],
            df['nearest_error_fraction']], showfliers=False)
ax.legend(['1 = qlearning_intelligent', '2 = qlearning_greedy', '3 = qlearning_epsilon', '4 = qlearning_epsilon_decay', '5 = expected_sarsa', '6 = expected_sarsa_decay', '7 = greedy', '8 = nearest'], loc='upper left', handletextpad=0, handlelength=0)
plt.ylabel('error fraction')
plt.xlabel('dataset')
plt.savefig('stats_states_derivation.png', dpi=300)
plt.show()
plt.close(fig)

# comparison of error derivation only for qlearning
fig, ax = plt.subplots()
ax.xaxis.grid()
ax.yaxis.grid()
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.5, 3)
ax.boxplot([df['qlearning_intelligent_error_fraction'],
            df['qlearning_greedy_error_fraction'],
            df['qlearning_epsilon_error_fraction'],
            df['qlearning_epsilon_decay_error_fraction'],
            df['expected_sarsa_error_fraction']], showfliers=False)
ax.legend(['1 = qlearning_intelligent', '2 = qlearning_greedy', '3 = qlearning_epsilon', '4 = qlearning_epsilon_decay', '5 = expected_sarsa'], loc='upper left', handletextpad=0, handlelength=0)
plt.ylabel('error fraction')
plt.xlabel('dataset')
plt.savefig('stats_states_derivation_qlearning.png', dpi=300)
plt.show()
plt.close(fig)

# regression plots
fig, ax = plt.subplots()
sns.regplot(df['qlearning_intelligent_states_percentage'],
            df['qlearning_intelligent_error_fraction'], order=2, ci=100, x_bins=10, x_estimator=np.mean, ax=ax)
sns.regplot(df['qlearning_greedy_states_percentage'],
            df['qlearning_greedy_error_fraction'], order=2, ci=100, x_bins=10, x_estimator=np.mean, ax=ax)
sns.regplot(df['qlearning_epsilon_states_percentage'],
            df['qlearning_epsilon_error_fraction'], order=2, ci=100, x_bins=10, x_estimator=np.mean, ax=ax)
sns.regplot(df['qlearning_epsilon_decay_states_percentage'],
            df['qlearning_epsilon_decay_error_fraction'], order=2, ci=100, x_bins=10, x_estimator=np.mean, ax=ax)
sns.regplot(df['expected_sarsa_states_percentage'],
            df['expected_sarsa_error_fraction'], order=2, ci=100, x_bins=10, x_estimator=np.mean, ax=ax)
plt.savefig('stats_regression_qlearning.png', dpi=300)
plt.show()
plt.close(fig)

print("Error Max:")
print(df.filter(regex='.*_error_fraction', axis=1).max())
print("")

print("Q-Learning Intelligent Difference:", 1 - (df[df['qlearning_intelligent_error_fraction'] <= 0.0].count()[0] / count))
print("Q-Learning Greedy Difference:", 1 - (df[df['qlearning_greedy_error_fraction'] <= 0.0].count()[0] / count))
print("Q-Learning Epsilon Difference:", 1 - (df[df['qlearning_epsilon_error_fraction'] <= 0.0].count()[0] / count))
print("Q-Learning Decay Difference:", 1 - (df[df['qlearning_epsilon_decay_error_fraction'] <= 0.0].count()[0] / count))
print("Expected SARSA Difference:", 1 - (df[df['expected_sarsa_error_fraction'] <= 0.0].count()[0] / count))
print("Expected SARSA Decay Difference:", 1 - (df[df['expected_sarsa_decay_error_fraction'] <= 0.0].count()[0] / count))
print("Greedy Difference:", 1 - (df[df['greedy_error_fraction'] <= 0.0].count()[0] / count))
print("Nearest Difference:", 1 - (df[df['nearest_error_fraction'] <= 0.0].count()[0] / count))
