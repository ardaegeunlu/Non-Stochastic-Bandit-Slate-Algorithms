# -*- coding: utf-8 -*-


import numpy as np
import HelperAlgorithms as helpers
import MultiplicativeWeights as mw
import math


class UnorderedSlatesBandit:

    def __init__(self, number_of_actions, slate_size, max_rounds):
        """
        :param number_of_actions: number of actions from which the slates will be formed, K.
        :param slate_size: slate size, s.
        :param max_rounds: the number of rounds for which the algorithm will run.
        """
        initial_dist = np.full(number_of_actions, 1.0 / number_of_actions) # uniform distribution.

        self.slate_size = slate_size
        self.number_of_actions = number_of_actions

        self.round = 0
        self.max_round = max_rounds # set as a constant

        self.gamma = math.sqrt((1.0 * number_of_actions/slate_size) * math.log(1.0*number_of_actions/ slate_size)/ (1.0*self.max_round))
        self.eta = math.sqrt( 1.0* (1.0 - self.gamma)* slate_size * math.log(1.0*number_of_actions/slate_size) / (1.0*number_of_actions * self.max_round ))
        self.regret_bound = 4.0 * math.sqrt(
            1.0 * slate_size * number_of_actions * max_rounds * math.log(1.0 * number_of_actions / slate_size))

        self.mw_engine = mw.MultiplicativeWeights(initial_dist, self.slate_size, self.eta)

        self.environment = None
        self.loss_vs_rounds = np.zeros(max_rounds, dtype=float)

    def set_environment(self, environment):
        """
        :param environment: this should be a function that can take a vector of size K
        (indicator vector of the chosen slate), and the current round, t as parameters and return the loss/reward
        associated with that slate and that slate only. The indicator vector will have non-zero elements which represent
        the chosen actions in that slate and zero elements which represent actions that are not chosen. The reward/loss
        for actions that are not chosen must be 0, and for the chosen actions the reward/loss should be in [-1,1] or else
        it will be clipped. Hence the output vector must also be a vector of size K with elements clipped to be in [-1,1].
        """
        self.environment = environment


    def iterate_agent(self):
        """
        run the agent for "max_rounds" rounds
        """
        for self.round in range(0, self.max_round):

            # get current prob. distribution from multiplicative weights object.
            current_distribution = np.copy(self.mw_engine.distribution)
            intermediate_dist = current_distribution * (1.0 - self.gamma) + (1.0*self.gamma / self.number_of_actions ) * np.ones(self.number_of_actions)

            # decompose the intermediate probability distribution into slate-indicator vectors and their corresponding probabilities.
            list_of_slate_prob_pairs = helpers.mixture_decomposition(self.slate_size, self.number_of_actions, intermediate_dist )

            # build prob array
            prob_array = []
            for pair in list_of_slate_prob_pairs:
                prob_array = np.append(prob_array, pair.probability)

            # choose a random slate using the probability array
            chosen_slate_index = np.random.choice(list_of_slate_prob_pairs.__len__(), None , replace=False, p=prob_array)

            chosen_slate = list_of_slate_prob_pairs[chosen_slate_index].indicator

            # using the environment function get the loss vector for the chosen slate, and push the loss to mw algorithm

            loss_vector = self.environment(chosen_slate, self.round)
            np.clip(loss_vector, -1, 1, loss_vector) # clip the loss to fit into [-1,1]

            # record the loss
            round_loss = loss_vector.sum()
            self.loss_vs_rounds[self.round] = round_loss

            # push to mw.
            loss_vector = loss_vector/(self.slate_size * intermediate_dist)
            self.mw_engine.computeAfterLossUnordered(loss_vector)

            self.round = self.round + 1



