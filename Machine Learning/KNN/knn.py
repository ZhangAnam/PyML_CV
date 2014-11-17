from __future__ import division
import numpy as np
import operator

class Distance:
    def dist(self, X, Y):
        raise NotImplemented

class  EuclideanDistance(Distance):
    def dist(self, X, Y):
        return sum((X - Y) ** 2) ** 0.5

def classify_knn(inX, dataset, label, k, dist=EuclideanDistance()):
    distances = np.array([dist.dist(i, inX) for i in dataset])
    sorted_dist = distances.argsort()
    vote = {}
    for i in range(k):
        vote[label[sorted_dist[i]]] = vote.get(label[sorted_dist[i]], 0) + 1
    winner = sorted(vote.iteritems(), key=operator.itemgetter(1), reverse=True)
    return winner[0][0]

def data_from_file(filename):
    f = open(filename)
    try:
        text = f.readlines()
        arraySize = len(text)
        fristLine = text[0].strip()
        elementSize = len(fristLine.split('\t'))
        mat = np.zeros((arraySize,elementSize))
        index = 0
        for line in text:
            line = line.strip()
            elements = line.split('\t')
            mat[index,:] = elements[:]
            index +=1
    finally:
        f.close()
    return mat

def data_norm(dataset):
    minVal = dataset.min(0)
    maxVal = dataset.max(0)
    ranges = maxVal - minVal
    #normDataSet = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDataSet = dataset - np.tile(minVal,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet , ranges , minVal

if __name__ == "__main__":
    data = data_from_file("datingTestSet2.txt")
    testRatio = 0.10
    m = data.shape[0]
    testNum = int(m*testRatio)
    dataset_train , ranges , minVal = data_norm(data[0:m-testNum,0:3])
    dataset_test = (data[m-testNum:m,0:3] - np.tile(minVal,(testNum,1))) / np.tile(ranges,(testNum,1))
    label_train = data[0:m-testNum,3]
    label_test = data[m-testNum:m,3]
    label_learn = np.array([classify_knn(inX, dataset_train, label_train, k=3) for inX in dataset_test])
    trueValue = 0
    for i in range(testNum):
        print("the classfiy value %d , real value %d" % (label_learn[i] , label_test[i]))
        if(label_learn[i] == label_test[i]):
            trueValue += 1
    print("the total error is %f" % (1 - trueValue/testNum))
    
