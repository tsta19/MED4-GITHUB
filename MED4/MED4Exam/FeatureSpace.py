import numpy as np

class FeatureSpace:
    printData = False


    prevState = 0

    featuresNum = 5


    def setMethods(self, methods):
        self.methods = methods

    def setEmotions(self, emotions):
        self.emotions = emotions
        self.resetMoodScores()

    def setFeatureSpaces(self):
        print("Getting imput from list")

        f = open("Feature Values/featurespacevariables.txt", "r")
        input = f.read()
        print(input)
        lines = input.replace("[", "").replace("]", "").strip().split('\n')
        print(len(lines))
        print(lines)
        features = [[0.0 for i in range(self.featuresNum)] for ii in range(len(lines) - 2)]
        print(features)
        for i in range(2, len(lines)):
            arr = lines[i].split(", ")
            for x in range(len(arr)):
                features[i - 2][x] = float(arr[x])

        print(features)
        print("Done getting input from list")

        if self.methods[0]:
            self.pitchMean = []
            self.pitchStd = []

            index = 0
            for x in range(len(self.emotions)):
                self.pitchMean.append(features[index][0])
                self.pitchStd.append(features[index+1][0])
                index += 2

        if self.methods[2]:
            self.pitchVariMean = []
            self.pitchVariStd = []

            index = 0
            for x in range(len(self.emotions)):
                self.pitchVariMean.append(features[index][0])
                self.pitchVariStd.append(features[index+1][0])
                index += 2
        if self.methods[3]:
            self.soundVariMean = []
            self.soundVariStd = []

            index = 0
            for x in range(len(self.emotions)):
                self.soundVariMean.append(features[index][0])
                self.soundVariStd.append(features[index+1][0])
                index += 2
        if self.methods[1]:
            self.soundLevelMean = []
            self.soundLevelStd = []

            index = 0
            for x in range(len(self.emotions)):
                self.soundLevelMean.append(features[index][0])
                self.soundLevelStd.append(features[index+1][0])
                index += 2
        if self.methods[4]:
            self.pwrFreqMean = []
            self.pwrFreqStd = []

            index = 0
            for x in range(len(self.emotions)):
                self.pwrFreqMean.append(features[index][0])
                self.pwrFreqStd.append(features[index+1][0])
                index += 2

    def getDistance(self, featureMeasurement, featureValue):
        distance = np.sqrt((featureValue - featureMeasurement) ** 2)

        return distance

    def getRelation(self, mean, std, measurementsArrayValue):
        #relation = (mean + self.getDistance(measurementsArrayValue, mean)) / (mean + std)
        relation = self.zeroDivision((mean + self.getDistance(measurementsArrayValue, mean)), (mean + std))
        return relation

    def zeroDivision(self, n, d):
        return n / d if d > 0 and n > 0 else 10000

    def checkPitch(self, measurementsArray):
        self.p = []

        for x in range(len(self.emotions)):
            self.p.append(self.getRelation(self.pitchMean[x], self.pitchStd[x], measurementsArray[0]))

        if self.printData:
            print('---------------------------')
            for x in range(len(self.emotions)):
                print("Pitch " + self.emotions[x] + ":", self.p[x])

        mostProbableMood = [min(self.p), self.p.index(min(self.p))]
        return mostProbableMood

    def checkPitchVariance(self, measurementsArray):
        index = 2 + self.minusCounter
        self.pVar = []

        for x in range(len(self.emotions)):
            self.pVar.append(self.getRelation(self.pitchVariMean[x], self.pitchVariStd[x], measurementsArray[index]))

        if self.printData:
            print('---------------------------')
            for x in range(len(self.emotions)):
                print("Pitch Variance " + self.emotions[x] + ":", self.pVar[x])

        mostProbableMood = [min(self.pVar), self.pVar.index(min(self.pVar))]
        return mostProbableMood

    def checkSoundVariance(self, measurementsArray):
        index = 3 + self.minusCounter
        self.sVar = []

        for x in range(len(self.emotions)):
            self.sVar.append(self.getRelation(self.soundVariMean[x], self.soundVariStd[x], measurementsArray[index]))

        if self.printData:
            print('---------------------------')
            for x in range(len(self.emotions)):
                print("Sound Variance " + self.emotions[x] + ":", self.sVar[x])

        mostProbableMood = [min(self.sVar), self.sVar.index(min(self.sVar))]
        return mostProbableMood

    def resetMoodScores(self):
        self.score, self.value = [0 for x in range(len(self.emotions))], [0 for x in range(len(self.emotions))]

    def checkSound(self, measurementsArray):
        index = 1 + self.minusCounter
        self.s = []

        for x in range(len(self.emotions)):
            self.s.append(self.getRelation(self.soundLevelMean[x], self.soundLevelStd[x], measurementsArray[index]))

        mostProbableMood = [min(self.s), self.s.index(min(self.s))]

        if self.printData:
            print('---------------------------')
            for x in range(len(self.emotions)):
                print("Sound Level " + self.emotions[x] + ":", self.s[x])

        return mostProbableMood

    def checkMostPowerfulFrequency(self, measurementsArray):
        index = 4 + self.minusCounter
        self.pF = []

        for x in range(len(self.emotions)):
            self.pF.append(self.getRelation(self.pwrFreqMean[x], self.pwrFreqStd[x], measurementsArray[index]))

        mostProbableMood = [min(self.pF), self.pF.index(min(self.pF))]

        if self.printData:
            print('---------------------------')
            for x in range(len(self.emotions)):
                print("Powerful Frequency " + self.emotions[x] + ":", self.pF[x])

        return mostProbableMood

    def checkEmotion(self, measurementsArray):
        if self.printData:
            print("measurements: " + str(measurementsArray))

        self.minusCounter = 0

        if self.methods[0]:
            pitch = self.checkPitch(measurementsArray)

            for x in range(len(self.emotions)):

                if pitch[1] == x and pitch[0] < 1:
                    self.value += pitch[0]
                    self.score[x] += 1
                    if self.printData:
                        print(f"Score[{x}] awarded from pitch level")
                        break
        else:
            self.minusCounter -= 1

        if self.methods[1]:
            soundlvl = self.checkSound(measurementsArray)

            for x in range(len(self.emotions)):

                if soundlvl[1] == x and soundlvl[0] < 1:
                    self.value += soundlvl[0]
                    self.score[x] += 1
                    if self.printData:
                        print(f"Score[{x}] awarded from pitch level")
                        break
        else:
            self.minusCounter -= 1

        if self.methods[2]:
            pitchVariance = self.checkPitchVariance(measurementsArray)

            for x in range(len(self.emotions)):
                if pitchVariance[1] == x and pitchVariance[0] < 1:
                    self.value += pitchVariance[0]
                    self.score[x] += 1
                    if self.printData:
                        print(f"Score[{x}] awarded from pitch level")
                        break
        else:
            self.minusCounter -= 1

        if self.methods[3]:
            soundVariance = self.checkSoundVariance(measurementsArray)

            for x in range(len(self.emotions)):
                if soundVariance[1] == x and soundVariance[0] < 1:
                    self.score[x] += 1
                    self.value += soundVariance[0]
                    if self.printData:
                        print(f"Score[{x}] awarded from pitch level")
                        break
        else:
            self.minusCounter -= 1

        if self.methods[4]:
            powerFrequency = self.checkMostPowerfulFrequency(measurementsArray)

            for x in range(len(self.emotions)):

                if powerFrequency[1] == x and powerFrequency[0] < 1:
                    self.value += powerFrequency[0]
                    self.score[x] += 1
                    if self.printData:
                        print(f"Score[{x}] awarded from pitch level")
                        break
        else:
            self.minusCounter -= 1

        theEmotionArray = []

        for x in range(len(self.emotions)):
            theEmotionArray.append(self.zeroDivision(self.value[x], self.score[x]))


        for x in range(len(self.emotions)):
            if self.score[x] > 1:
                theEmotionArray[x] = theEmotionArray[x] * 0.70
                if self.score[x] > 2:
                    theEmotionArray[x] = theEmotionArray[x] * 0.70
                    if self.score[x] > 3:
                        theEmotionArray[x] = theEmotionArray[x] * 0.50

        if self.printData:
            print('---------------------------')

        arr = [d > 1 for d in theEmotionArray]
        if self.printData:
            print("index af mindste i emotion array: "+str(theEmotionArray.index(min(theEmotionArray))))
            print("emotion array min: "+str(min(theEmotionArray)))

            print('---------------------------')

            for x in range(len(self.emotions)):
                print(f"Score {self.emotions[x]}: " + str(self.score[x]))
            print('---------------------------')
            for x in range(len(self.emotions)):
                print(f"Emotion value {self.emotions[x]}: " + str(theEmotionArray[x]))
            print('---------------------------')

        if (all(arr)):
            print("No emotion detected, staying in same state")
            self.resetMoodScores()
            return 4

        if theEmotionArray.index(min(theEmotionArray)) == 0:
            print("Happy")
            self.prevState = 0
            self.resetMoodScores()
            return 0

        elif theEmotionArray.index(min(theEmotionArray)) == 1:
            print("Sad")
            self.prevState = 1
            self.resetMoodScores()
            return 1

        elif theEmotionArray.index(min(theEmotionArray)) == 2:
            print("Angry")
            self.prevState = 2
            self.resetMoodScores()
            return 2

        elif theEmotionArray.index(min(theEmotionArray)) == 3:
            print("Fear")
            self.prevState = 3
            self.resetMoodScores()
            return 3

        if self.printData:
            print("--------------------------------------------------------")
            print("---------------- THE END OF ANALYSIS -------------------")
            print("--------------------------------------------------------")
