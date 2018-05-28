# -*- coding: utf-8 -*-

import numpy as np
"""
    This class contains test environments for bandit algorithms with ordered and unordered slates.
    Test cases for both of the algorithms are same. 
    
    ->For the unordered case, each arm is associated with a normal probability distribution with a mean between
    [-0.4, 0.4] and std = 0.3. The mean is selected at start from a uniform random distribution from the range of
    [-0.4, 0.4]. One can use the get_blind_loss(slate_indicator, round) as environment function.
    
    ->For the ordered case, each arm is again associated with a normal probability distribution with a mean between
    [-0.4, 0.4] and std = 0.3. The mean is selected at start from a uniform random distribution from the range of
    [-0.4, 0.4]. First action chosen has a multiplier of 1, second action chosen has a multiplier of 1.1, third has a multiplier of 1.2
    and so on. One can use the get_blind_loss_ordered(slate_indicator_matrix, round) as environment function.
"""


class Environment:

    def __init__(self, number_of_actions, round_no, slate_size):

        """
        :param number_of_actions: K.
        :param round_no: max_rounds, time horizon.
        :param slate_size: s.
        """
        self.slate_size = slate_size

        self.number_of_actions = number_of_actions
        self.distribution_means = (np.random.rand(number_of_actions)) - 0.5
        self.distribution_means = self.distribution_means / 1.25

        self.best_slate_loss = 0
        print("\n\nDistribution Means: {0}".format(self.distribution_means))
        self.loss_matrix = np.zeros([round_no, number_of_actions])

        for i in range(0, round_no) :
            self.loss_matrix[i,:] = self.get_loss_normal()

        action_rewards = np.sum(self.loss_matrix, axis=0)

        self.best_slate = action_rewards.argsort()[0:slate_size]
        self.best_slate_vs_rounds = np.sum(self.loss_matrix[:,self.best_slate],axis=1)

        ordered_loss = self.loss_matrix[:, self.best_slate] * (np.arange(1, slate_size + 1)[::-1])
        np.clip(ordered_loss, -1, 1, ordered_loss)

        self.best_slate_vs_rounds_ordered = np.sum(ordered_loss, axis=1)
        

    def get_blind_loss(self, slate_indicator, round_no):

        loss_vector = np.copy(self.loss_matrix[round_no, :])
        loss_vector[slate_indicator == 0] = 0
        return loss_vector

    def get_blind_loss_ordered(self, subpermutation_matrix, round_no):

        blind_matrix = np.zeros(subpermutation_matrix.shape)
        for position in range (0, self.slate_size):
            
            loss_vector = np.copy(self.loss_matrix[round_no, :])
            loss_vector[subpermutation_matrix[position,:] == 0] = 0
            blind_matrix[position, :] = loss_vector * (1 + (position)/10.0)

        np.clip(blind_matrix, -1, 1, blind_matrix)
        return blind_matrix


    def get_loss_normal(self):

        loss_vector = np.zeros(self.number_of_actions)
        for i in range(0, self.number_of_actions):
            loss_vector[i] = np.random.normal(self.distribution_means[i], 0.3)

        np.clip(loss_vector, -1, 1, loss_vector)

        return loss_vector









