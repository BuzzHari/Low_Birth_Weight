import pandas as pd
import numpy as np
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

def calc_priors(train_arr):
    priors = {}
    for row in train_arr:
        if row[-1] not in priors:
            priors[row[-1]] = 1
        else:
            priors[row[-1]] += 1
    priors[0] = priors[0]/float(len(train_arr))
    priors[1] = priors[1]/float(len(train_arr))
    return priors


def calc_pdf(x, mean, stdev):
    if stdev == 0.0:
        if x == mean:
            return 1.0
        else:
            return 0.0
    e = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev,2))))
    return 1 / (math.sqrt(2 * math.pi) * stdev) * e

def calc_class_prob(mean_and_std,priors,row):
    probs = {}
    for (class_value, mean_std) in mean_and_std.items():
        #Intializing it with priori porb..ie P(yi)
        probs[class_value] = priors[class_value];
        #print(class_value, probs[class_value], len(mean_std),len(mean_and_std[0]), len(mean_and_std[1]))
        for i in range(len(mean_std)):
            (mean, stdev) = mean_std[i]
            x = row[i]
            prob = calc_pdf(x, mean, stdev)
            if prob > 0:
                probs[class_value] += math.log(prob)
    return probs

def predict(mean_and_std, priors, row):
    probs = calc_class_prob(mean_and_std, priors, row)
    bestLabel = None
    bestProb = -1
    for (class_value, prob) in probs.items():
        if bestLabel is None or prob > bestProb:
            bestProb = prob
            bestLabel = class_value
    return bestLabel

def get_pred(mean_and_std, priors, test_arr):
    preds = list()
    for row in test_arr:
        res = predict(mean_and_std, priors, row)
        preds.append(res)
    return preds

def gnb(train_arr, test_arr):

    print("Train_Arr", len(train_arr))
    mean_and_std = calc_mean_and_std(train_arr)
    priors = calc_priors(train_arr)
    print(priors)
    pred = get_pred(mean_and_std, priors, test_arr)
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
    n_splits = 2
    accu = 0
    avg_acc = 0
    test_seed = 0
    for seed in range(10000):
        kf = KFold(n_splits, shuffle = True, random_state = seed)
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
        avg_acc += mean_accu
        print("Mean_accu: %.3f%%" % mean_accu)

        if(mean_accu > accu):
            accu = mean_accu
            test_seed = seed
    print("Accu:", accu)
    print("Test_seed", test_seed)
    print("Avg:", avg_acc/10000)


