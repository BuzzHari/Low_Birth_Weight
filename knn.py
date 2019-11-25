import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from math import sqrt

def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)-1):
        distance += (point1[i] - point2[i])**2
    return sqrt(distance)

def nearest_neighbours(test_row, train_arr, no_of_neighbours):
    distances = list()
    for row in train_arr:
        dist = euclidean_distance(test_row, row)
        distances.append((row, dist))

    distances.sort(key=lambda x: x[1])
    neighbours = list()
    for i in range(no_of_neighbours):
        neighbours.append(distances[i][0])
    return neighbours


def knn(train_arr, test_arr, no_of_neighbours):
    predictions = list()
    for row in test_arr:
        neighbours = nearest_neighbours(row, train_arr, no_of_neighbours)
        class_values = [x[-1] for x in neighbours]
        predictions.append(max(set(class_values), key = class_values.count))

    return(predictions)

def calc_accuracy(actual_values, pred_values):
    count = 0
    for i in range(len(actual_values)):
        if actual_values[i] == pred_values[i]:
            count += 1
    return count / float(len(actual_values)) * 100.0

if __name__ == '__main__':

    LBW_Data = 'CleanedDataSet.csv'
    dataFrame = pd.read_csv(LBW_Data)
    #features_X = np.array(dataFrame.drop('Reuslt', axis = 1))
    #labels_y = np.array(dataFrame['Reuslt'])

    train_arr= np.array(dataFrame)

    n_splits = 5
    kf = KFold(n_splits,shuffle = True, random_state = 42)
    folds = list()
    for train_index, test_index in kf.split(train_arr):
        train, test = train_arr[train_index], train_arr[test_index]
        folds.append((train,test))


    no_of_neighbours = 80

    accu = 0
    for i in range(1,no_of_neighbours):
        result = list()
        for fold in folds:
            train_arr = fold[0]
            test_arr = fold[1]

            pred_values = knn(train_arr, test_arr, i)
            actual_values = [x[-1] for x in test_arr]
            accuracy = calc_accuracy(actual_values, pred_values)
            result.append(accuracy)

        mean_accu = sum(result)/float(len(result))
        if(mean_accu > accu):
            accu = mean_accu
            optimal_neigh = i
        print("Neighbours:",i)
        print("Scores: %s" % result)
        print("Mean Accuracy: %.3f%%" % mean_accu )

    print("Optimal Neighbour", optimal_neigh)
