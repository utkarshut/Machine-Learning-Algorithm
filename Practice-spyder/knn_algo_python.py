#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:56:52 2019

@author: utkarsh
"""

import csv
import random
with open('./dataSet/iris/iris.csv') as csvFile:
    lines = csv.reader(csvFile)
    for row in lines:
        True
        #print(', '.join(row))

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename,'r') as csvFile:
        lines =csv.reader(csvFile)
        next(lines, None)  # skip the headers
        dataset= list(lines)
        for x in range(len(dataset)-1):
            dataset[x]= dataset[x][1:]
            for y in range(0,4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                print(x, dataset[x])
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                
trainingSet = []
testSet = []
loadDataset('./dataSet/iris/iris.csv',0.66,trainingSet,testSet)

#print('Train:'+repr(len(trainingSet)))
#print('Train:'+repr(len(testSet)))

import math

def euclideanDistance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance += pow ((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

data1 = [2,2,3,'a']
data2 = [4,4,4,'b']

distance = euclideanDistance(data1,data2,3)

#print('Distance'+repr(distance))


import operator

def getNeighbors(trainingSet,testInstance,k):
    distances =[]
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


trainSet = [[2,2,3,'a'],[4,4,4,'b']]
testInstance = [3,3,3]

k = 1

neighbors = getNeighbors(trainSet,testInstance,1)
#print("neighbors ",neighbors)


import operator

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

neighbors = [[1,1,1,'a'],[2,2,2,'b'],[3,3,3,'b']]
response =getResponse(neighbors)
#print(response)

def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


testSet = [[1,1,1,'a'],[2,2,3,'a'],[3,3,3,'b']]
predictions = ['a','a','a']
accuracy = getAccuracy(testSet , predictions)
#print(accuracy)


def main():
    trainingSet =[]
    testSet = []
    split = 0.67
    loadDataset('./dataSet/iris/iris.csv',0.66,trainingSet,testSet)
    predictions = []
    k = 3
    print(trainingSet)
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet,testSet[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted ='+repr(result)+' ,actual=' +repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet,predictions)
    print('accuracy:'+repr(accuracy)+'%')
        
main()