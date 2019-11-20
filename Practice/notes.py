#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:20:41 2019

@author: utkarsh
"""

result = getResponse(neighbors)
        prediction.append(result)
        print('> predicted ='+repr(result)+' ,actual=' +repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet,prdictions)
    print('accuracy:'+repr(accuracy)+'%')
    
    
    
    neighbors = getNeighbors(trainingSet,testSet[x],k)
        result = getResponse(neighbors)
        prediction.append(result)
        print('> predicted ='+repr(result)+' ,actual=' +repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet,prdictions)
    print('accuracy:'+repr(accuracy)+'%')[3.1, '5.4', 2.1, 'Iris-virginica']