from sklearn import model_selection
import numpy as np
import os

class Evaluation():

    def __init__(self, fe):
        self.fe = fe
        self.fs = self.fe.getFeatureSpace()

    def ExtractSoundFiles(self, emotion, noiseRange, voiceRange, chunkRange):
        emotionArr = np.array([])
        np.set_printoptions(suppress=True)
        folder = "Emotions/" + emotion + "/"
        for filename in os.listdir(folder):
            if filename is not None:
                tempArr = self.fe.get_features_from_clip(folder, filename, noiseRange, voiceRange, chunkRange)
                if len(tempArr) > 1:
                    if len(emotionArr) == 0:
                        emotionArr = tempArr
                    else:
                        emotionArr = np.vstack((emotionArr, tempArr))
        f = open("Feature Values/" + str(emotion) + ".txt", "w")
        emotionArr = emotionArr.tolist()
        for x in range(len(emotionArr)):
            f.write(str(emotionArr[x]) + "\n")
        f.close()
        emotionArr = np.array(emotionArr)
        print()
        print(emotionArr)
        print(len(emotionArr[0]))
        print(len(emotionArr))

        return emotionArr

    def makeDatasetFromText(self, emotions):
        self.featuresX = np.array([])
        self.featuresY = np.array([])
        for i in range(len(emotions)):
            f = open("Feature Values/" + str(emotions[i]) + ".txt", "r")
            readlines = f.read()
            lines = readlines.replace("[", "").replace("]", "").strip().split('\n')

            features = [line.split(', ') for line in lines]

            for x in range(len(features)):
                for y in range(len(features[x])):
                    features[x][y] = float(features[x][y])

            features = self.cut_off_array(features)

            for x in range(len(features)):
                if len(self.featuresX) <= 1:
                    self.featuresX = features[x]
                else:
                    self.featuresX = np.vstack((self.featuresX, features[x]))
            self.featuresY = np.append(self.featuresY, np.array([i for x in range(len(features))]))

        print()
        print("dataset made")

    def makeDatasetFromSound(self, emotions, noiseRange, voiceRange, chunkRange):
        self.featuresX = np.array([])
        self.featuresY = np.array([])
        for i in range(len(emotions)):
            features = self.ExtractSoundFiles(emotions[i], noiseRange, voiceRange, chunkRange)

            for x in range(len(features)):
                if len(self.featuresX) <= 1:
                    self.featuresX = features[x]
                else:
                    self.featuresX = np.vstack((self.featuresX, features[x]))
            self.featuresY = np.append(self.featuresY, np.array([i for x in range(len(features))]))

        print()
        print("dataset made")

    def train(self, emotions, methods):
        self.fs.setMethods(methods)
        self.fs.setEmotions(emotions)

        featuresX = [[i for e, i in enumerate(self.featuresX[x]) if methods[e] is True] for x in range(len(self.featuresX))]
        print("featureX: " + str(featuresX))

        x_train, x_test, y_train, y_test = model_selection.train_test_split(featuresX, self.featuresY, test_size=0.2)

        numOfMethods = np.count_nonzero(methods)
        print(numOfMethods)

        arr = [[[] for x in range(numOfMethods)] for i in range(len(emotions))]

        print(arr)

        print(x_train)
        print(y_train)

        for x in range(len(x_train)):
            for y in range(numOfMethods):
                arr[int(y_train[x])][y].append(x_train[x][y])

        f = open("Feature Values/featurespacevariables.txt", "w")
        f.write(str(emotions)+"\n")
        f.write(str(methods)+"\n")
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

        return [ytest, predictResult]


