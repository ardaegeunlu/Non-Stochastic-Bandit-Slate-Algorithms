# -*- coding: utf-8 -*-

import math
import numpy as np
import sys
import CornerProbabilityPack as cp

"""
    This file includes the following algorithms, which are employed to fulfill several functionalities required by 
    the bandit algorithm with unordered and ordered slates.
    
    1. Mixture Decomposition for unordered slates. 
    2. Mixture Decomposition for ordered slates without states.
    3. Mixture Decomposition for ordered slates with states.
    4. Capping Algorithm.
    5. Cyclic Bregman Projection.
    6. Cartesian Product.
    
"""
def mixture_decomposition(slate_size, total_number_of_actions, vector_to_be_decomposed):

    """
    :param slate_size: the slate size, s.
    :param total_number_of_actions: all of the possible actions that slates can be formed from, K.
    :param vector_to_be_decomposed: the scaled probability vector that the agent currently has.
    :return: a list of slate indicator vectors and associated probabilities, a list of CornerProbabilityPack.
    """

    # this is the implementation of the "mixture decomposition algorithm" in the paper
    # Randomized Online PCA Algorithms with Regret Bounds that are Logarithmic in the Dimension
    # by Warmuth et al. 2008.

    # please refer to it for details.

    list_of_cornerprob_pairs = []
    target_vector = np.copy(vector_to_be_decomposed)
    # the elements of the corner vectors(indicator vectors) will need to sum to 1.
    corner_element = 1.0/(slate_size)

    # if the sum of the elements is too little, we can abort the algorithm as the probabilities associated
    # with corner vectors from there on would be too small to create any difference.
    while np.sum(target_vector) > 10 ** (-9):

        corner = np.zeros(total_number_of_actions)
        current_size = np.sum(target_vector)
        # all decomposed corner vectors need to contain all of the elements of magnitude |target_element|
        target_element = current_size * 1.0 / slate_size

        # list of all the indices.
        all_indices = np.arange(total_number_of_actions)
        # get all elements that equal target_element
        essential_indices = np.where((np.absolute(target_vector - target_element) < (10**-12)))
        # list non-zero indices in the target vector
        zero_indices = np.where(target_vector == 0.0)
        # get all indices that are both non-zero and not a target_element
        remaining_indices = np.setdiff1d(all_indices, essential_indices)
        remaining_indices = np.setdiff1d(remaining_indices, zero_indices)

        # if the number of the target_elements is less than slate_size, fill the corner vector with elements from
        # remaining_indices
        if slate_size - np.size(essential_indices) > 0:
            complemantary_indices = np.random.choice(remaining_indices, (slate_size - np.size(essential_indices)), False)
            chosen_indices = np.append(essential_indices, complemantary_indices)
        else:
            chosen_indices = essential_indices

        # all of the unchosen indices are set to zero.
        unchosen_vector = np.copy(target_vector)
        unchosen_vector[chosen_indices] = 0
        corner[chosen_indices] = corner_element

        l_value = np.amax(unchosen_vector)
        s_value = np.amin(target_vector[chosen_indices])
        # probability associated with this indicator vector.
        p_value = (min(slate_size * s_value , np.sum(target_vector) - slate_size*l_value))

        # if the probability that is paired with this vector is too small, we can ignore it
        if  p_value > 10**(-10):
            pack = cp.CornerProbabilityPack(corner, p_value)
            list_of_cornerprob_pairs.append(pack)

        # prob < 0 should never happen.
        elif p_value < 0 and math.fabs(p_value) > 10**-8:
            print("Corner- prob pair {0} & {1}".format(corner, p_value))
            sys.exit("ERROR, prob of corner < 0 !")

        # update the target vector
        target_vector = target_vector - np.multiply(corner, p_value)


    return list_of_cornerprob_pairs

def mixture_decomposition_ordered_stateless(slate_size, total_number_of_actions, matrix_to_be_decomposed):

    """
    :param slate_size: slate size, s.
    :param total_number_of_actions: number of actions that the slate will be formed from, K.
    :param matrix_to_be_decomposed: the scaled probability matrix of position, action pairs.
    :return: if successful: list of decomposed matrix-probability pairs, else: False.
    """
    # this algorithm uses the ideas in mixture decomposition and try to generalize them to a matrix.
    # - It is required that in every decomposition there is an action selected for every so called position,
    # and all of the actions in the slate are unique i.e. the sum of all elements in each&every row should be 1/s
    # and the sum of all elements in each&every column should also be 1/s.

    list_of_cornerprob_pairs = []
    target_matrix = np.copy(matrix_to_be_decomposed)

    consecutive_fails = 0
    cumulative_prob_sum = 0

    corner_element = 1.0 / (slate_size)
    column_number = total_number_of_actions / slate_size
    column_condition = np.full(shape=column_number, fill_value=corner_element)
    row_condition = np.full(slate_size, fill_value=corner_element)

    while np.sum(target_matrix) > 10 ** (-9):

        corner = np.zeros(total_number_of_actions)

        current_size = np.sum(target_matrix)
        target_element = current_size * 1.0 / slate_size

        all_indices = np.arange(total_number_of_actions)
        essential_indices = np.where((np.absolute(target_matrix - target_element) < (10 ** -8)))
        zero_indices = np.where(target_matrix == 0.0)
        remaining_indices = np.setdiff1d(all_indices, essential_indices)
        remaining_indices = np.setdiff1d(remaining_indices, zero_indices)

        if slate_size - np.size(essential_indices) > 0:
            complemantary_indices = np.random.choice(remaining_indices, (slate_size - np.size(essential_indices)),
                                                     False)
            chosen_indices = np.append(essential_indices, complemantary_indices)
        else:
            chosen_indices = essential_indices


        unchosen_vector = np.copy(target_matrix)
        unchosen_vector[chosen_indices] = 0
        corner[chosen_indices] = corner_element

        reshapedCorner = corner.reshape(slate_size, column_number)
        column_sums = reshapedCorner.sum(axis=0)
        row_sums = reshapedCorner.sum(axis=1)

        areSelectionConditionsSatisfied = (np.all(column_sums <= column_condition) and np.all(row_sums == row_condition))

        # if the row and column conditions are satisfied, proceed
        if areSelectionConditionsSatisfied:

            l_value = np.amax(unchosen_vector)
            s_value = np.amin(target_matrix[chosen_indices])
            p_value = (min(slate_size * s_value, np.sum(target_matrix) - slate_size * l_value))

            if p_value > 10 ** (-10):
                consecutive_fails = 0
                # print(corner)
                pack = cp.CornerProbabilityPack(corner, p_value)
                list_of_cornerprob_pairs.append(pack)
                target_matrix = target_matrix - np.multiply(corner, p_value)
                cumulative_prob_sum += p_value
            elif p_value < 0 and math.fabs(p_value) > 10 ** -8:
                print("Corner- prob pair {0} & {1}".format(corner, p_value))
                sys.exit("ERROR, prob of corner < 0 !")

            else:
                consecutive_fails += 1
        # if column and row conditions are not satisfied, this decomposition was useless, try again to decompose from
        # where we have left.
        else:
            consecutive_fails += 1

        # enter if the attempts to decompose the matrix fail too many times consecutively
        if consecutive_fails > 50:

            # already created enough decompositions to add up to more than 0.85 probability,
            # add a stub decomposition to cover the remaining probability,
            # if the player chooses the stub as the next slate, it will try to pick another slate.
            if cumulative_prob_sum > 0.85:
                # print("Returning pack with stub & cum sum was {0}".format(cumulative_prob_sum))
                corner = 'STUB'
                p_value= 1.0 - cumulative_prob_sum
                pack = cp.CornerProbabilityPack(corner, p_value)
                list_of_cornerprob_pairs.append(pack)
                return list_of_cornerprob_pairs

            # if the cumulative probability of decompositions were less than 0.85
            # restart the decomposition from the start.

            #print("too many consecutive fails, trying to decompose again .. ")
            #print("cumulative prob was: {0}".format(cumulative_prob_sum))
            return False

    #print("\nFOUND A FULL DECOMPOSITION\n")
    return list_of_cornerprob_pairs


def mixture_decomposition_ordered(slate_size, total_number_of_actions, matrix_to_be_decomposed):

    """
    :param slate_size: slate size, s.
    :param total_number_of_actions: total number of actions from which the slates will be formed, K.
    :param the scaled probability matrix of position, action pairs.
    :return: if successful: list of decomposed matrix-probability pairs, else: False.
    """
    # this algorithm is just like the algorithm above, but it tries to explore the space of all possible decompositions
    # with an order. If the algorithm above fails to create a decomposition matrix that will satisfy
    # the row&column conditions it just randomly tries to decompose again
    # but it never backtracks to change older decompositions that would ultimately affect current decompositions.
    # This algo. pushes all possible decompositions that could have been randomly selected as well to a stack, pops and uses one,
    # and if it fails at any step, it pops the one on top and tries to go on. This gives the algorithm a way to backtrack
    # and explore the space with more order but it is often stuck at bad decompositions for a long time due to its nature.

    # This algorithm is currently much slower than the decomposer which does not keep track of other possible decompositions
    # and states. But there is room for future improvement for speeding it up,
    # like shuffling decompositions before pushing them in etc.

    list_of_cornerprob_pairs = []
    stack_of_unused_corners = [] # stack of all possible decompositions
    stack_of_number_of_unused_corners = [] # stack of #of unused decompositions

    target_matrix = np.copy(matrix_to_be_decomposed)

    consecutive_fails = 0
    cumulative_prob_sum = 0

    corner_element = 1.0 / (slate_size)
    isGoingForward = True # start with a forward step.

    while np.sum(target_matrix) > 10 ** (-9):

        readyToAddPack = True
        corner = np.zeros((slate_size, total_number_of_actions))

        current_size = np.sum(target_matrix)
        target_element = current_size * 1.0 / slate_size

        # is this is a forward step?
        if isGoingForward:
            all_indices = np.arange(total_number_of_actions)
            all_chosen_indices =[]

            for row in range(0, slate_size):
                essential_indices = np.where((np.absolute(target_matrix[row,:] - target_element) < (10 ** -8)))
                zero_indices = np.where(target_matrix[row,:] == 0.0)
                remaining_indices = np.setdiff1d(all_indices, essential_indices)
                remaining_indices = np.setdiff1d(remaining_indices, zero_indices)

                if np.size(essential_indices) < 1:

                    # failed to decompose properly, break
                    if remaining_indices.size < 1:
                        isGoingForward = False
                        readyToAddPack = False
                        break

                    complemantary_indices = np.random.choice(remaining_indices, remaining_indices.size,
                                                     False)
                    chosen_indices = complemantary_indices
                else:
                    chosen_indices = essential_indices

                all_chosen_indices.append(chosen_indices)

            # did not break out of the loop above, go here
            if isGoingForward:

                all_possible_combinations = cartesian(all_chosen_indices)
                rows_to_be_deleted = []
                for row in range(0, all_possible_combinations.shape[0]):

                    if (np.unique(all_possible_combinations[row, :]).size != slate_size):
                       rows_to_be_deleted.append(row)

                all_possible_combinations = np.delete(all_possible_combinations, rows_to_be_deleted, axis=0)

                if all_possible_combinations.shape[0] > 0:

                    unchosen_vector = np.copy(target_matrix)
                    used_indices = all_possible_combinations[0, :]
                    s_candidates = np.empty(slate_size)
                    for i in range(0, slate_size):
                        unchosen_vector[i, used_indices[i]] = 0
                        corner[i, used_indices[i]] = corner_element
                        s_candidates[i] = target_matrix[i, used_indices[i]]

                    all_possible_combinations = np.delete(all_possible_combinations, 0, axis=0)
                    [remaining_options, garbage] = all_possible_combinations.shape

                    for k in range(0, remaining_options):
                        stack_of_unused_corners.append(all_possible_combinations[0, :])
                    stack_of_number_of_unused_corners.append(remaining_options)

                    l_value = np.amax(unchosen_vector)
                    s_value = np.amin(s_candidates)
                    p_value = (min(slate_size * s_value, np.sum(target_matrix) - slate_size * l_value))

                else:
                    isGoingForward = False
                    readyToAddPack = False

        # backtrack
        else:

            #print("went a step back")
            howManyOptions = stack_of_number_of_unused_corners.pop()
            consecutive_fails += 1
            while howManyOptions < 1:
                consecutive_fails += 1
                cornerPack = list_of_cornerprob_pairs.pop()
                prob = cornerPack.probability
                permMatrix = cornerPack.indicator
                cumulative_prob_sum -= prob
                target_matrix = target_matrix + np.multiply(permMatrix, prob)
                if stack_of_number_of_unused_corners.__len__() > 0:
                    howManyOptions = stack_of_number_of_unused_corners.pop()
                else:
                    return False

            howManyOptions -= 1
            stack_of_number_of_unused_corners.append(howManyOptions)
            used_indices = stack_of_unused_corners.pop()

            unchosen_vector = np.copy(target_matrix)
            s_candidates = np.empty(slate_size)

            for i in range(0, slate_size):
                unchosen_vector[i, used_indices[i]] = 0
                corner[i, used_indices[i]] = corner_element
                s_candidates[i] = target_matrix[i, used_indices[i]]

            l_value = np.amax(unchosen_vector)
            s_value = np.amin(s_candidates)
            p_value = (min(slate_size * s_value, np.sum(target_matrix) - slate_size * l_value))

        if p_value > 10 ** (-10) and readyToAddPack:

            pack = cp.CornerProbabilityPack(corner, p_value)
            list_of_cornerprob_pairs.append(pack)
            consecutive_fails = 0
            isGoingForward = True
            target_matrix = target_matrix - np.multiply(corner, p_value)
            cumulative_prob_sum += p_value

            # print("not wasted loop")
        elif p_value < 0 and math.fabs(p_value) > 10 ** -8:
            print("Corner- prob pair {0} & {1}".format(corner, p_value))
            sys.exit("ERROR, prob of corner < 0 !")

        else:
            consecutive_fails += 1
            isGoingForward = False

        # stuck with a bad decomposition
        if consecutive_fails > 500:
            # if already above cum. prob > 0.80 add stub and return
            if cumulative_prob_sum > 0.80 and cumulative_prob_sum < 1.0:
                # print("Returning pack with stub & cum sum was {0}".format(cumulative_prob_sum))
                corner = 'STUB'
                p_value = 1.0 - cumulative_prob_sum
                pack = cp.CornerProbabilityPack(corner, p_value)
                list_of_cornerprob_pairs.append(pack)
                return list_of_cornerprob_pairs

            # otherwise, restart decomposition
            # print("too many consecutive fails, trying to decompose again .. ")
            # print("cumulative prob was: {0}".format(cumulative_prob_sum))
            return False


    print("\nFOUND A FULL DECOMPOSITION\n")
    return list_of_cornerprob_pairs


def capping_algorithm(target_vector, slate_size):

    # please refer to Warmuth et al. 2008 for the details of this algorithm.
    """
    :param target_vector: vector to be subjected to bregman projection.
    :param slate_size: slate size, s.
    :return: vector that can be used by multiplicative weights.
    """
    # sorted indices of the target_vector in decreasing order
    sorting_indices = np.argsort(target_vector)[::-1]
    n = target_vector.size
    # already capped, return.
    if target_vector.max() <= 1.0 / slate_size:
        return target_vector

    i = 1
    temp_w = np.full(n, 10)

    # Set first i largest components to 1/s and normalize the rest to (s-i)/s while max(vector) > 1/s
    while temp_w.max() > (1.0 / slate_size) :

        temp_w = np.copy(target_vector)
        temp_w[sorting_indices[0:i]] = 1.0 / slate_size

        sum_of_remaining_elements = (temp_w[sorting_indices[i:]]).sum()

        (temp_w[sorting_indices[i : ]]) = ( 1.0 * (1.0 * slate_size - i) / slate_size ) * (temp_w[sorting_indices[i:]]) / sum_of_remaining_elements

        i = i + 1

    return temp_w

def cyclic_bregman_projection(target_matrix):

    # please refer to Kale et al. 2010 for the details of this algorithm.
    """
    :param target_matrix: matrix to be projected
    :return: projected matrix that can be used by multiplicative weights.
    """
    [row_number, col_number] = target_matrix.shape

    lambda_vector = np.ones(col_number)
    previous_state = np.zeros(target_matrix.shape)

    k = 0

    while True:

        if (np.absolute(target_matrix - previous_state)).sum() <= 10**-13:
            break;
        previous_state = np.copy(target_matrix)

        # row phase -- iterate over all rows, and rescale them to make
        # them sum to 1/s.

        row_scaling_factors = ((1.0*target_matrix.sum(axis=1)*row_number)[:,np.newaxis])
        target_matrix = target_matrix / row_scaling_factors

        # column phase -- for every col. compute scaling factor alpha to
        # rescale col. to 1/s.

        col_scaling_factors = (1.0 * target_matrix.sum(axis=0) * row_number)
        scaling_factors = np.minimum(lambda_vector, 1.0/col_scaling_factors)
        lambda_vector = lambda_vector / scaling_factors
        target_matrix = target_matrix * scaling_factors

        k += 1

    return target_matrix

def cartesian(arrays, out=None):

    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    #>>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

