#! /usr/bin/python3

import pandas
import numpy as np
from tree_node import TreeNode

training_dataframe = pandas.read_csv("./titanic_data/train.csv")

def makePredictors(frame):
    return [ frame["Pclass"].tolist(), [1 if s == "male" else 0 for s in frame["Sex"]], frame["Age"].tolist() ]

def makeResponses(frame):
    return frame["Survived"].tolist()

baseline_survival_rate = np.mean(training_dataframe["Survived"])
predictions_from_baseline = [1.0 if baseline_survival_rate > 0.5 else 0.0 for i in training_dataframe["Survived"]]
error_rate_from_baseline = np.mean([abs(x-y) for (x, y) in zip(predictions_from_baseline, training_dataframe["Survived"])])
print("Baseline survival rate: {}".format(baseline_survival_rate))
print("Accuracy from baseline prediction: {}".format(1.0 - error_rate_from_baseline))

K = 4
print_costs_during_training=False

training_accuracies = []
validation_accuracies = []

for k in range(K):
    training_fold   = training_dataframe.select(lambda i: i%K != k)
    validation_fold = training_dataframe.select(lambda i: i%K == k)

    training_predictors = makePredictors(training_fold)
    training_responses = makeResponses(training_fold)


    tn = TreeNode(training_predictors, training_responses)
    if print_costs_during_training: print("Cost: {}".format(tn.get_cost()))
    while tn.enact_best_split():
        if print_costs_during_training: print("Cost: {}".format(tn.get_cost()))
    
    print("Fold {}:".format(k))

    training_predictions = [tn.predict(x) for x in zip(training_predictors[0], training_predictors[1], training_predictors[2])]
    training_errors = [0.0 if abs(x-y) < 0.5 else 1.0 for (x,y) in zip(training_predictions, training_responses)]
    training_accuracy = 1.0 - np.mean(training_errors)

    print("In-sample accuracy: {}".format(training_accuracy))
    training_accuracies.append(training_accuracy)

    validation_predictors = makePredictors(validation_fold)
    validation_responses = makeResponses(validation_fold)

    validation_predictions = [tn.predict(x) for x in zip(validation_predictors[0], validation_predictors[1], validation_predictors[2])]
    validation_errors = [0.0 if abs(x-y) < 0.5 else 1.0 for (x,y) in zip(validation_predictions, validation_responses)]
    validation_accuracy = 1.0 - np.mean(validation_errors)
    print("Validation accuracy: {}".format(validation_accuracy))
    validation_accuracies.append(validation_accuracy)

print("  ***  ***  ***  ***  ***  ***  ***")
print("Mean training accuracy:   {}".format(np.mean(training_accuracies)))
print("Mean validation accuracy: {}".format(np.mean(validation_accuracies)))
