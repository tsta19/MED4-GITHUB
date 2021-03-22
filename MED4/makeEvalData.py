import numpy as np
import random
from QuickSort import *

# This should not be in the final build :D

outF = open("testEvalData.txt","w")
counter = 0

def printResults(lowValx, lowValy, highValy, type, counterStart):
    counter = counterStart
    for x in range(50):
        rndArrSize = random.randrange(50, 100)
       # print(rndArrSize)

        low = round(random.uniform(lowValx,lowValy),4)
        high = round(random.uniform(lowValy,highValy),4)

        arr = [0]*rndArrSize


        for i in range(rndArrSize):
            arr[i] = round(random.uniform(low,high),4)

        qs = QuickSort()
        arr = qs.quickSort(arr)

        string = str(type)
        for i in range(rndArrSize):
            string += ", " + str(arr[i])

        print(string)
        outF.write(string)

        counter+=1
        if counter != 100:
            outF.write("\n")

printResults(46.0,65.0,89.0,0,0)
printResults(60.0,80.0,120.0,1,50)


outF.close()
