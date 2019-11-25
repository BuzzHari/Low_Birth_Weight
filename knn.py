import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from math import sqrt
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)-1):
        distance += (point1[i] - point2[i])**2
    return sqrt(distance)

def weightDistances(no_of_neighbours, distances):
    weights = np.zeros(no_of_neighbours, dtype=np.float32)
    total = 0.0
    for i in range(no_of_neighbours):
        weights[i] += 1.0/distances[i]
        total += weights[i]
    weights /= total
    return weights

def nearest_neighbours(test_row, train_arr, no_of_neighbours):
    distances = list()
    for row in train_arr:
        dist = euclidean_distance(test_row, row)
        distances.append((row, dist))

    distances.sort(key=lambda x: x[1])
    #print("Len of total Distance:", len(distances))
    #for i in range(len(distances)):
    #    print(distances[i][1])
    neighbours = list()
    for i in range(no_of_neighbours):
        neighbours.append(distances[i])
    return neighbours

def call_vote(weightDists, neighbours):
    votes = np.zeros(2, dtype=np.float32)
    class_values = [x[-1] for x in neighbours]
    #print(class_values)
    for i in range(len(weightDists)):
        pred_class = class_values[i]
        votes[int(pred_class)] += weightDists[i] * 1.0
        #print("Votes:")
        #print(votes)
    if(votes[0] > votes[1]):
        return 0
    else:
        return 1

def knn(train_arr, test_arr, no_of_neighbours):
    predictions = list()
    #print(len(test_arr))
    for row in test_arr:
        neighbours = list()
        distances = np.zeros(no_of_neighbours, dtype=np.float32)
        weightDists = np.zeros(no_of_neighbours, dtype=np.float32)

        neighbours_dis = nearest_neighbours(row, train_arr, no_of_neighbours)
        for i in range(no_of_neighbours):
            distances[i] = neighbours_dis[i][1]
            neighbours.append(neighbours_dis[i][0])
        weightDists = weightDistances(no_of_neighbours, distances)
        #print("WeightDistnaces:")
        #print(weightDists)
        #class_values = [x[-1] for x in neighbours]
        #predictions.append(max(set(class_values), key = class_values.count))
        predictions.append(call_vote(weightDists, neighbours))
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
    kf = KFold(n_splits,shuffle = True, random_state = 7)
    folds = list()
    for train_index, test_index in kf.split(train_arr):
        train, test = train_arr[train_index], train_arr[test_index]
        folds.append((train,test))


    no_of_neighbours = 3

    accu = 0
    graph = list()
    neighbours = list()
    for i in range(3,no_of_neighbours+1):
        result = list()
        for fold in folds:
            train_arr = fold[0]
            test_arr = fold[1]

            pred_values = knn(train_arr, test_arr, i)
            actual_values = [x[-1] for x in test_arr]
            accuracy = calc_accuracy(actual_values, pred_values)
            result.append(accuracy)

        mean_accu = sum(result)/float(len(result))
        graph.append(mean_accu)
        neighbours.append(i)
        if(mean_accu > accu):
            accu = mean_accu
            optimal_neigh = i
        #print("Neighbours:",i)
        print("Scores: %s" % result)
        print("Mean Accuracy: %.3f%%" % mean_accu )

    #print("Optimal Neighbour", optimal_neigh)
    #print(graph)
    #print(neighbours)
    #plt.plot(neighbours,graph)
    #plt.xlabel("K")
    #plt.ylabel("Accuracy")
    #plt.title("Elbow Curve")
    #plt.show()
