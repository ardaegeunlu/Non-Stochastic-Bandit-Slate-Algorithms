# Non-Stochastic-Bandit-Slate-Problems
Implementations of the bandit algorithms with unordered and ordered slates that are described in the paper "Non-Stochastic Bandit Slate Problems", by Kale et al. 2010.

We consider bandit problems, motivated by applications in online advertising and news story selection, in which the learner must repeatedly select a slate, that is, a subset of size s from K possible actions, and then receives rewards for just the
selected actions. The goal is to minimize the regret with respect to total reward of the best slate computed in hindsight.¹

In the unordered version of the problem, the reward to the learning algorithm is the sum of the rewards of the chosen actions in the slate. So the chosen actions all have a weight of 1 and therefore they are “unordered”. 

In the ordered slate problem, the adversary specifies a reward for using an action in a specific position. The reward to the learner then is the sum of the rewards of the (actions, position) pairs in the chosen ordered slate.

Sample code to run the bandit algorithms. More can be found in the folder TestEnvironment.
```python
import UnorderedSlatesBandit as unordered
import OrderedSlatesBandit as ordered
import numpy as np
import math

# use either the ordered or unordered bandit.
agent = unordered.UnorderedSlatesBandit(actions, slate_size, rounds)
# agent = ordered.OrderedSlatesBandit(actions, slate_size, rounds)

# set up the environment function which will give rewards to player.
# environment function should take a slate indicator vector or slate-position indicator matrix as input
# and return a loss/reward vector/matrix of equal dimensions of input.
agent.set_environment(environment.get_blind_loss)

# run the agent for 'rounds' times.
agent.iterate_agent()

# environment is responsible of storing best_fixed_slate_data
print("\nagent reward: {0} vs best slate reward: {1}".format(-np.sum(agent.loss_vs_rounds), -   np.sum(environment.best_slate_vs_rounds)))
```
