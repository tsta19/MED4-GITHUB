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
            self.featuresY = np.append(self.featuresY, np.array([i for x in range(len(features))]))

    def train(self, emotions):
        print(self.featuresX)
        print(len(self.featuresX))
        print(self.featuresY)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(self.featuresX, self.featuresY, test_size=0.2)
        print(x_train)
        print(x_test)
        print(y_train)
        print(y_test)

        arr = [[0for ii in range(5)] for i in range(len(emotions))]

        arr = np.asarray(arr)
        for x in range(len(x_train)):
            if len(arr[int(y_train[x])]) <= 5:
                arr[int(y_train[x])] = x_train[x]
            else:
                for i in range(5):
                    print(i)
                    arr[int(y_train[x])][i] = np.append(arr[int(y_train[x])][i], x_train[x][i])

        f = open("featurespacevariables.txt", "w")
        f.write(str(emotions)+"\n")
        f.write("Pitch, Sound Level, Pitch Variance, Sound Level Variance, Power Frequency\n")

        for x in range(len(arr)):
            featureSpace = [np.mean(arr[x][0]), np.mean(arr[x][1]), np.mean(arr[x][2]), np.mean(arr[x][3]), np.mean(arr[x][4])]
            print("Feature Mean: " + str(featureSpace))
            featureSTD = [np.std(arr[x][0]), np.std(arr[x][1]), np.std(arr[x][2]), np.std(arr[x][3]), np.mean(arr[x][3])]
            print("Feature STD: " + str(featureSTD))
            f.write(str(featureSpace) + "\n")
            f.write(str(featureSTD) + "\n")

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
        print(confusion_matrix(y_test, predictResult))

        return [y_test, predictResult]


    # Funktion som returnerer en string an på hvilket 'target' det har.
    def target_type(self, row):
        if row['Target'] == 1:
            return 'Voice'
        else:
            return 'Noise'

    def find_errors(self, testY, predictions, testX):
        for i in range(len(testY)):
            if testY[i] != predictions[i]:
                print("error with [" + str(testX[i]) + "][" + str(testY[i]) + "]")



e = Evaluation()
emotions = ["Angry", "Fear", "Happiness", "Sad"]
e.makeDataset(emotions)

conMatrixArr = []

iterations = 100

for x in range(iterations):
    x_test, y_test = e.train(emotions)

    checkList = [0, 1, 2, 3]

    counter = 0
    for y in range(len(checkList)):

        if checkList[y] in y_test:
            counter += 1

    if counter > len(checkList)-1:
        conMatrixArr.append(e.test(x_test, y_test))

print("------------------------------")
print()

print(len(conMatrixArr))

accuracy = np.array([])

for x in range(len(conMatrixArr)):
    accuracy = np.append(accuracy, accuracy_score(conMatrixArr[x][0], conMatrixArr[x][1]))
    print("Accuracy: " + str(accuracy[x]))
    print("Confusion Matrix:")
    print(confusion_matrix(conMatrixArr[x][0], conMatrixArr[x][1]))
    print()

print("Mean Accuracy: " + str(np.mean(accuracy)))

