import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score
import numpy as np
import cv2
import os
import wave
from FeatureExtraction import FeatureExtraction


class Evaluation:
    fe = FeatureExtraction()
    fs = fe.getFeatureSpace()

    def ExtractSoundFiles(self, emotion):
        emotionArr = np.array([])
        folder = "Sound_Files/Emotions/" + emotion + "/"
        for filename in os.listdir(folder):
            if filename is not None:
                tempArr = self.fe.get_features_from_clip(folder, filename)
                print("tempArr: " + str(tempArr))
                if len(tempArr) != 0:
                    if len(emotionArr) == 0:
                        emotionArr = tempArr
                    else:
                        emotionArr = np.vstack((emotionArr, tempArr))
                        # i in range(len(tempArr)):
                            #emotionArr = np.vstack(tempArr[i])
        return emotionArr

    def makeDataset(self, emotions):
        self.featuresX = np.array([])
        self.featuresY = np.array([])
        for i in range(len(emotions)):
            features = self.ExtractSoundFiles(emotions[i])
            for x in range(len(features)):
                if len(self.featuresX) <= 1:
                    self.featuresX = features[x]
                else:
                    self.featuresX = np.vstack((self.featuresX, features[x]))
                print(self.featuresX)
            self.featuresY = np.append(self.featuresY, np.array([i for x in range(len(features))]))

        print()
        print("dataset made")

    def train(self, emotions, methods):
        self.fs.setMethods(methods)
        
        print(self.featuresX)
        print(len(self.featuresX))
        print(self.featuresY)

        for i in enumerate(self.featuresX[0]):
            print(i)

        #featuresX = [x for x in self.featuresX[] if methods[x] is True]
        #featuresX = [[y for y in self.featuresX[x] if methods[self.featuresX.index(y)] is True]for x in range(len(self.featuresX))]
        featuresX = [[i for e, i in enumerate(self.featuresX[x]) if methods[e] is True]for x in range(len(self.featuresX))]
        print(featuresX)
        print("featureX: " + str(featuresX))

        x_train, x_test, y_train, y_test = model_selection.train_test_split(featuresX, self.featuresY, test_size=0.2)
        print(x_train)
        print(x_test)
        print(y_train)
        print(y_test)

        numOfMethods = np.count_nonzero(methods)
        print(numOfMethods)

        arr = [[[] for x in range(numOfMethods)] for i in range(len(emotions))]
        print("arrrrrrr" + str(arr))

        for x in range(len(x_train)):
            for y in range(numOfMethods):
                arr[int(y_train[x])][y].append(x_train[x][y])
            print(arr)

        f = open("featurespacevariables.txt", "w")
        f.write(str(emotions)+"\n")
        f.write("Pitch, Sound Level, Pitch Variance, Sound Level Variance, Power Frequency\n")
        print(arr)
        for x in range(len(arr)):
            featureSpace = np.array([])
            featureSTD = np.array([])
            for i in range(numOfMethods):
                print(arr[x][i])
                featureSpace = np.append(featureSpace, np.mean(arr[x][i]))
                featureSTD = np.append(featureSTD, np.std(arr[x][i]))

            print("Feature Mean: " + str(featureSpace))
            print("Feature STD: " + str(featureSTD))
            f.write(str(featureSpace.tolist()) + "\n")
            f.write(str(featureSTD.tolist()) + "\n")

        f.close()

        self.fs.setFeatureSpaces()

        return x_test, y_test

    def test(self, xtest, ytest):
        predictResult = np.array([])
        print(xtest)
        for x in range(len(xtest)):
            predictResult = np.append(predictResult, self.fs.checkEmotion(xtest[x]))

        accuracy = accuracy_score(ytest, predictResult)

        print("Accuracy: " + str(accuracy))
        print("Confusion Matrix:")
        print(confusion_matrix(ytest, predictResult))

        return [ytest, predictResult]


