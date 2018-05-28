# -*- coding: utf-8 -*-

import numpy as np
import HelperAlgorithms as helpers
import MultiplicativeWeights as mw
import math


class OrderedSlatesBandit:

    def __init__(self, number_of_actions, slate_size, max_rounds):
        """
        :param number_of_actions: number of actions from which the slates will be formed, K.
        :param slate_size: slate size, s.
        :param max_rounds: the number of rounds for which the algorithm will run.
        """
        initial_dist = np.full((slate_size, number_of_actions), 1.0 / (slate_size*number_of_actions), dtype=float)

        self.slate_size = slate_size
        self.number_of_actions = number_of_actions

        self.round = 0
        self.max_round = max_rounds # set as a constant

        self.regret_bound = 4.0 *slate_size* math.sqrt(
            1.0 * number_of_actions * max_rounds * math.log(1.0 * number_of_actions ))

        self.gamma = math.sqrt(1.0 * number_of_actions * math.log(1.0*number_of_actions) / max_rounds)
        self.eta = math.sqrt((1.0-self.gamma) * math.log(1.0*number_of_actions)/(1.0*number_of_actions * max_rounds))
        self.mw_engine = mw.MultiplicativeWeights(initial_dist, self.slate_size, self.eta)

        self.environment = None
        self.loss_vs_rounds = np.zeros(max_rounds, dtype=float)

    def set_environment(self, environment):
        """
        :param environment: this should be a function that can take a matrix of shape (s,K)
        (indicator matrix of the chosen slate of position, action pairs), and the current round, t as parameters
        and return the loss/reward associated with that slate and that slate only.
        The indicator matrix will have non-zero elements which represent the chosen actions in that slate and zero
        elements which represent actions that are not chosen. The reward/loss
        for actions that are not chosen must be 0, and for the chosen actions the reward/loss should be in [-1,1] or else
        it will be clipped. Hence the output matrix must also be a matrix of shape (s,K) with elements clipped to be in [-1,1].
        """
        self.environment = environment


    def iterate_agent(self):
        """
        run the agent for "max_rounds" rounds
        """
        for self.round in range(0, self.max_round ):

            # get the current distribution from mw.
            current_distribution = np.copy(self.mw_engine.distribution)

            intermediate_dist = current_distribution * (1.0 - self.gamma)\
                                + (1.0*self.gamma / (self.number_of_actions * self.slate_size) ) * np.ones((self.slate_size,self.number_of_actions))

            # flatten distribution
            flattened_dist = intermediate_dist.flatten(order='C')
            list_of_slate_prob_pairs = helpers.mixture_decomposition_ordered_stateless(self.slate_size, self.number_of_actions* self.slate_size, flattened_dist  )
            # repeat if it fails
            while (list_of_slate_prob_pairs == False):
                list_of_slate_prob_pairs = helpers.mixture_decomposition_ordered_stateless(self.slate_size,
                                                                                 self.number_of_actions*self.slate_size,
                                                                                                          flattened_dist)
            # build the prob array associated with slate matrices.
            prob_array = []
            for pair in list_of_slate_prob_pairs:
                prob_array = np.append(prob_array, pair.probability)

            # choose 2 slates in case one of them is a stub.
            [chosen_slate_index1, chosen_slate_index2] = np.random.choice(list_of_slate_prob_pairs.__len__(), size=2 , replace=False, p=prob_array)
            chosen_slate = list_of_slate_prob_pairs[chosen_slate_index1].indicator
            if chosen_slate == 'STUB':
                chosen_slate = list_of_slate_prob_pairs[chosen_slate_index2].indicator


            chosen_slate = np.reshape(chosen_slate, (self.slate_size, self.number_of_actions))

            #using the environment class get the loss vector for the chosen slate, and push the loss to mw algorithm
            #to get new distribution

            loss_vector = self.environment(chosen_slate, self.round)

            round_loss = loss_vector.sum()
            self.loss_vs_rounds[self.round] = round_loss

            loss_vector = loss_vector / (self.slate_size * intermediate_dist)
            self.mw_engine.computeAfterLossOrdered(loss_vector)

            self.round = self.round + 1

            if (self.round % 100 == 0):
                print("@round: {0}".format(self.round))


