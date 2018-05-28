# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import UnorderedSlatesBandit as unordered
import OrderedSlatesBandit as ordered

import numpy as np
import math
import Environment as env

def initialize_normal_dist_environment():
    environment = env.Environment(actions, rounds, slate_size)
    return environment

# parameters for bandit
total_runs = 5 # number of trials
actions = 7 # total number of actions from which slates will be formed, K.
slate_size = 3 # also noted as "s".
rounds = 5000 # number of rounds at each run.

agent_loss_vs_rounds = np.zeros(rounds)
best_slate_loss_vs_rounds = np.zeros(rounds)

for i in range(0,total_runs):

    # use either the ordered or unordered bandit.
    agent = unordered.UnorderedSlatesBandit(actions, slate_size, rounds)
    # agent = ordered.OrderedSlatesBandit(actions, slate_size, rounds)

    # set up the environment function which will give rewards to player.
    # environment function should take a slate indicator vector or slate-position indicator matrix as input.
    # and return a loss/reward vector of equal dimensions of input.
    environment = initialize_normal_dist_environment()
    agent.set_environment(environment.get_blind_loss)

    # run the agent for 'rounds' times.
    agent.iterate_agent()

    agent_loss_vs_rounds = agent_loss_vs_rounds + agent.loss_vs_rounds
    # environment is responsible for storing best_fixed_slate_data
    best_slate_loss_vs_rounds = best_slate_loss_vs_rounds + environment.best_slate_vs_rounds

    print("\nagent reward: {0} vs best slate reward: {1}".format(-np.sum(agent.loss_vs_rounds), -np.sum(environment.best_slate_vs_rounds)))

    most_preferred_actions = agent.mw_engine.distribution.argsort()[::-1]
    best_slate_indices = environment.best_slate
    print("best slate was {0} and most preferred actions were {1}".format(best_slate_indices, most_preferred_actions))

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ PLOTTING THE RESULTS ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

regret_bound = agent.regret_bound
print ("\nUnordered bandit slates run for {0} actions with slates of size {1} for {2} rounds for {3} times".format(actions, slate_size, rounds, total_runs))
print ("\nAgents avg. reward: {0}\nand best fixed slate's avg. reward: {1}".format(-agent_loss_vs_rounds.sum()/total_runs, -best_slate_loss_vs_rounds.sum()/total_runs))


avg_agent_cum_reward_vector = np.cumsum(-agent_loss_vs_rounds/total_runs)
avg_best_slate_cum_reward_vector = np.cumsum(-best_slate_loss_vs_rounds/total_runs)

plt.figure(0)
plt.plot(avg_agent_cum_reward_vector, "r-",label="agent reward")
plt.plot(avg_best_slate_cum_reward_vector, "g-", label= "best slate reward")

#unordered regret
regret_bound = 4.0 * np.sqrt(1.0*slate_size*actions*np.arange(0,rounds)*math.log(1.0*actions/slate_size))

#ordered regret
#regret_bound = 4.0 * slate_size * np.sqrt(1.0*actions*np.arange(0,rounds)*math.log(1.0*actions))

plt.plot(regret_bound, "b--", label= "regret bound")
regret = avg_best_slate_cum_reward_vector-avg_agent_cum_reward_vector
plt.plot(regret, "m--", label="regret")


plt.annotate('%0.2f' % avg_best_slate_cum_reward_vector[rounds - 1], xy=(1, avg_best_slate_cum_reward_vector[rounds - 1]), xytext=(8, 0),
    xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.annotate('%0.2f' % avg_agent_cum_reward_vector[rounds - 1], xy=(1, avg_agent_cum_reward_vector[rounds - 1]), xytext=(8, 0),
    xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.annotate('%0.2f' % regret_bound[rounds - 1], xy=(1, regret_bound[rounds - 1]), xytext=(8, 0),
    xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.annotate('%0.2f' % regret[rounds - 1], xy=(1, regret[rounds - 1]), xytext=(8, 0),
    xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.xlabel("rounds")
plt.ylabel("cumulative rewards")
plt.title("Unordered Bandit Slates, Best Fixed Slate vs Agent")
plt.legend()


plt.text(50,1000,"actions:{0}\nslates:{1}\nrounds:{2}\nruns:{3}".format(actions, slate_size, rounds, total_runs))

plt.figure(1)

# unordered regret vs ordered regret
plt.plot(regret_bound, "b--", label= "regret bound = 4*sqrt(T*s*K*ln(K/s))")
#plt.plot(regret_bound, "b--", label= "regret bound = 4*s*sqrt(K*ln(K)*T)")

regret = avg_best_slate_cum_reward_vector-avg_agent_cum_reward_vector
plt.plot(regret, "m--", label="regret")

plt.annotate('%0.2f' % regret_bound[rounds - 1], xy=(1, regret_bound[rounds - 1]), xytext=(8, 0),
    xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.annotate('%0.2f' % regret[rounds - 1], xy=(1, regret[rounds - 1]), xytext=(8, 0),
    xycoords=('axes fraction', 'data'), textcoords='offset points')

plt.xlabel("rounds")
plt.ylabel("cumulative regret")
plt.title("Unordered Bandit Slates, Regret")

plt.legend()
plt.show()



