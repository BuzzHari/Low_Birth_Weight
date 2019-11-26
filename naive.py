import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import math

def get_classes(train_arr):
    class_dic = {}
    for row in train_arr:
        if row[-1] not in class_dic:
            class_dic[row[-1]] = []
        class_dic[row[-1]].append(row)
    return class_dic

def calc(rows):
    mean_std = [(np.mean(x), np.std(x)) for x in zip(*rows)]
    mean_std.pop()
    return mean_std


def calc_mean_and_std(train_arr):
    dic = {}
    sep_classes = get_classes(train_arr)
    for (class_value, rows) in sep_classes.items():
        dic[class_value] = calc(rows)
    return dic

def calc_pdf(x, mean, stdev):
    if stdev == 0.0:
        if x == mean:
            return 1.0
        else:
            return 0.0
    e = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev,2))))
    return 1 / (math.sqrt(2 * math.pi) * stdev) * e

def calc_class_prob(mean_and_std, row):
    probs = {}
    for (class_value, mean_std) in mean_and_std.items():
        probs[class_value] = 0
        for i in range(len(mean_std)):
            (mean, stdev) = mean_std[i]
            x = row[i]
            prob = calc_pdf(x, mean, stdev)
            if prob > 0:
                probs[class_value] += math.log(prob)
    return probs

def predict(mean_and_std, row):
    probs = calc_class_prob(mean_and_std, row)
    bestLabel = None
    bestProb = -1
    for (class_value, prob) in probs.items():
        if bestLabel is None or prob > bestProb:
            bestProb = prob
            bestLabel = class_value
    return bestLabel

def get_pred(mean_and_std, test_arr):
    preds = list()
    for row in test_arr:
        res = predict(mean_and_std, row)
        preds.append(res)
    return preds

def gnb(train_arr, test_arr):

    mean_and_std = calc_mean_and_std(train_arr)
    pred = get_pred(mean_and_std, test_arr)
    return pred

def calc_accuracy(actual_values, pred_values):
    count = 0
    for i in range(len(actual_values)):
        if actual_values[i] == pred_values[i]:
            count += 1
    return count / float(len(actual_values)) * 100.0


if __name__ == "__main__":

    data = 'CleanedDataSet.csv'
    data = pd.read_csv(data)

    data = data.drop("History", axis = 1)

    data = data.to_numpy()
    n_splits = 4
    #for seed in range(10000):
    kf = KFold(n_splits, shuffle = True, random_state = 3508)
    folds = list()
    for train_index, test_index in kf.split(data):
        train, test = data[train_index], data[test_index]
        folds.append((train,test))

    results = list()
    for fold in folds:
        train_arr = fold[0]
        test_arr = fold[1]

        pred_values = gnb(train_arr, test_arr)
        actual_values = [x[-1] for x in test_arr]
        accuracy = calc_accuracy(actual_values, pred_values)
        results.append(accuracy)

    mean_accu =  sum(results)/float(len(results))
    print("Mean_accu: %.3f%%" % mean_accu)


