import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import cv2
import os
import wave
from FeatureExtraction import FeatureExtraction


class Evaluation:

    fe = FeatureExtraction()

    def getDataset(self, emotions):
        for x in range(len(emotions)):
            emotionArr = np.array([])
            folder = "Sound_Files/Emotions/" + emotions[x] + "/"
            for filename in os.listdir(folder):
                if filename is not None:
                    tempArr = self.fe.get_features_from_clip(folder,filename)
                    print(tempArr)
                    if len(tempArr[0]) != 0:
                        if len(emotionArr) == 0:
                            emotionArr = tempArr
                        else:
                            for i in range(4):
                                for ii in range(len(tempArr[i])):
                                    #print(str(i) + str(ii))
                                    emotionArr[i].append(tempArr[i][ii])
                #print(emotionArr)
            print(emotionArr)
            print()
            print(emotions[x])
            featureSpace = [np.mean(emotionArr[0]), np.mean(emotionArr[1]), np.mean(emotionArr[2]), np.mean(emotionArr[3])]
            print("Feature Mean: " + str(featureSpace))
            featureSTD = [np.std(emotionArr[0]), np.std(emotionArr[1]), np.std(emotionArr[2]), np.std(emotionArr[3])]
            print("Feature STD: " + str(featureSTD))
            print()


    #Funktion som returnerer en string an på hvilket 'target' det har.
    def target_type(self,row):
        if row['Target'] == 1:
            return 'Voice'
        else:
            return 'Noise'


    def find_errors(self,testY,predictions,testX):
        for i in range(len(testY)):
            if testY[i] != predictions[i]:
                print("error with [" + str(testX[i]) + "][" + str(testY[i]) + "]")

e = Evaluation()
e.getDataset(["Angry","Fear", "Happiness", "Sad", "Tender"])
