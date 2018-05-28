
# -*- coding: utf-8 -*-


import numpy as np
import HelperAlgorithms as helpers

"""
    This file includes the Multiplicative Weights algorithm that is essential
    to bandit algorithm for both unordered and ordered slates.
"""


class MultiplicativeWeights:

    def __init__(self, initial_distribution, slate_size, eta):
        """
        :param initial_distribution: specify the initial distribution over the actions
        :param slate_size: total number of actions in every slate
        :param eta: the constant eta
        """

        self.distribution = initial_distribution
        self.slate_size = slate_size
        self.eta = eta
        self.round = 1

        return

    def computeAfterLossUnordered(self, lossVector):
        """
        Use this only for unordered bandit slates
        :param lossVector: the loss vector in round t
        :return: void
        """

        temp1 = np.exp( np.multiply((-self.eta), lossVector))
        normalizationConst = np.dot(self.distribution, temp1)
        temp2 = ((temp1 * self.distribution) / normalizationConst)

        # now we need to project temp2 to set P using RE as a distance measure
        self.distribution = helpers.capping_algorithm(temp2, self.slate_size)

        return

    def computeAfterLossOrdered(self, lossVector):
        """
        Use this method for ordered bandit slates
        :param lossVector: the loss vector in round t
        :return: void
        """

        temp1 = np.exp(-self.eta * lossVector)
        normalizationConst = np.sum(temp1 * self.distribution)
        temp2 = ((temp1 * self.distribution) / normalizationConst)
        # now we need to project temp2 to set P using RE as a distance measure
        self.distribution = helpers.cyclic_bregman_projection(temp2)
        self.round += 1

        return

