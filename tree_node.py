import functools
import numpy as np

def uniquify_list(xs):
    return functools.reduce(lambda l, x: [x] if l == [] else l + [x] if x != l[-1] else l, xs, [])

def uniquify_index_list(xs):
    return functools.reduce(lambda l, x: [(x[0], [x[1]])] if l == [] else l + [(x[0], [x[1]])] if x[0] != l[-1][0] else l[:-1] + [(l[-1][0], l[-1][1] + [x[1]])], xs, [])

class CostTracker:
    def __init__(self, right_list):
        self.right_sum, self.right_sq_sum = functools.reduce(lambda acc, x: (acc[0] + x, acc[1] + x*x), right_list, (0.0, 0.0))
        self.left_sum, self.left_sq_sum = 0.0, 0.0
        self.right_N = len(right_list)
        self.left_N = 0

    def move_value_right_to_left(self, value):
        self.right_sum -= value
        self.right_sq_sum -= value*value
        self.left_sum += value
        self.left_sq_sum += value*value
        self.right_N -= 1
        self.left_N += 1

    def get_right_N(self):
        return self.right_N

    def get_left_N(self):
        return self.left_N

    def get_right_cost(self):
        if self.right_N == 0:
            return 0
        else:
            return self.right_sq_sum - self.right_sum*self.right_sum/self.right_N

    def get_left_cost(self):
        if self.left_N == 0:
            return 0
        else:
            return self.left_sq_sum - self.left_sum*self.left_sum/self.left_N

    def get_right_mean(self):
        return self.right_sum/self.right_N

    def get_left_mean(self):
        if self.left_N == 0:
            return None
        else:
            return self.left_sum/self.left_N

    def get_total_cost(self):
        if self.right_N == 0:
            return None
        else:
            return self.get_left_cost() + self.get_right_cost()


class TreeNode:
    def __init__(self, predictors, responses): 
        self.N = len(responses)
        self.p = len(predictors)
        for i in range(self.p):
            if (len(predictors[i]) != self.N):
                raise ValueError("List lengths differ. ({} responses, but predictor {} has {} values.)".format(self.N, i, len(predictors[i])))

        self.left_child = None
        self.right_child = None

        self.predictors = predictors
        self.responses = responses
        
        self.unsplit_MSE = np.var(predictors)

        # Make a sorted list of each predictor
        self.sorted_predictor_index_lists = [
            [x for x in zip(predictors[i], range(len(predictors[i])))] for i in range(self.p)
        ]
        [xs.sort() for xs in self.sorted_predictor_index_lists]

        self.left_right_list = [0 for i in range(self.N)]  # This will store whether a particular datum is in the left or right child tree

        self.best_costs = [None for i in range(self.p)]
        self.best_assignments = [None for i in range(self.p)]
        self.best_split_locations = [None for i in range(self.p)]
        self.best_prediction_pairs = [None for i in range(self.p)]

    def find_best_split(self, predictor_index):
        if predictor_index < 0 or predictor_index >= self.p:
            raise ValueError("predictor_index out of range ({}, should be between 0 and {} inclusive).".format(predictor_index, self.p-1))
        pred_values = self.sorted_predictor_index_lists[predictor_index]
        unique_pred_values = uniquify_index_list(pred_values)
        cost_tracker = CostTracker(self.responses)
        running_best_cost = cost_tracker.get_total_cost()

        # Everything starts off assigned to the right-hand split
        right_to_left_move_times = [self.N + 1 for x in self.predictors[predictor_index]]
        running_move_time = 0
        best_move_time = 0
        best_split_location = None
        best_left_prediction = None
        best_right_prediction = None

        # Walk through unique_pred_values, 'moving' from the right to left bucket one at a time
        for i in range(len(unique_pred_values)-1):
            indices_to_move = unique_pred_values[i][1]
            for j in indices_to_move:
                cost_tracker.move_value_right_to_left(self.responses[j])
                right_to_left_move_times[j] = running_move_time
            running_move_time += 1
            if cost_tracker.get_total_cost() < running_best_cost:
                # We've found a new best split
                running_best_cost = cost_tracker.get_total_cost()
                best_move_time = running_move_time
                best_split_location = 0.5*(unique_pred_values[i][0] + unique_pred_values[i+1][0])
                best_left_prediction = cost_tracker.get_left_mean()
                best_right_prediction = cost_tracker.get_right_mean()
         
        # Now cache the best split that exists for this predictor
        self.best_costs[predictor_index] = running_best_cost
        self.best_assignments[predictor_index] = ['L' if x < best_move_time else 'R' for x in right_to_left_move_times]
        self.best_split_locations[predictor_index] = best_split_location
        self.best_prediction_pairs[predictor_index] = (best_left_prediction, best_right_prediction)

    def report_best_split_cost(self):
        if self.left_child == None and self.right_child == None:
            # Ensure that we know the result of splitting on each of the indices
            for i in range(self.p):
                if self.best_costs[i] == None:
                    self.find_best_split(i)
            return min(self.best_costs)
        else:
            left_splitting_cost = self.left_child.report_best_split()
            right_splitting_cost = self.right_child.report_best_split()
            if left_splitting_cost == None:
                return right_splitting_cost
            elif right_splitting_cost == None:
                return left_splitting_cost
            else:
                return min(left_splitting_cost, right_splitting_cost)
