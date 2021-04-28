import numpy as np
import math


class FeatureSpace:
    hScore, hValue = 0, 0
    sScore, sValue = 0, 0
    aScore, aValue = 0, 0
    fScore, fValue = 0, 0
    tScore, tValue = 0, 0
    prevState = 0
    theEmotionScores = [hScore, sScore, aScore, fScore, tScore]

    def setFeatureSpaces(self):
        print("Getting imput from list")

        f = open("featurespacevariables.txt", "r")
        input = f.read()
        print(input)
        lines = input.replace("[", "").replace("]", "").strip().split('\n')
        print(len(lines))
        print(lines)
        features = [[0.0 for i in range(4)] for ii in range(len(lines) - 2)]
        print(features)
        for i in range(2, len(lines)):
            arr = lines[i].split(", ")
            for x in range(len(arr)):
                features[i - 2][x] = float(arr[x])

        print(features)
        print("Done getting input from list")

        self.pitchMean_happy = features[4][0]
        self.pitchStd_happy = features[5][0]
        self.pitchMean_sad = features[6][0]
        self.pitchStd_sad = features[7][0]
        self.pitchMean_angry = features[0][0]
        self.pitchStd_angry = features[1][0]
        self.pitchMean_fear = features[2][0]
        self.pitchStd_fear = features[3][0]

        self.pitchVariMean_happy = features[4][2]
        self.pitchVariStd_happy = features[5][2]
        self.pitchVariMean_sad = features[6][2]
        self.pitchVariStd_sad = features[7][2]
        self.pitchVariMean_angry = features[0][2]
        self.pitchVariStd_angry = features[1][2]
        self.pitchVariMean_fear = features[2][2]
        self.pitchVariStd_fear = features[3][2]

        self.soundVariMean_happy = features[4][3]
        self.soundVariStd_happy = features[5][3]
        self.soundVariMean_sad = features[6][3]
        self.soundVariStd_sad = features[7][3]
        self.soundVariMean_angry = features[0][3]
        self.soundVariStd_angry = features[1][3]
        self.soundVariMean_fear = features[2][3]
        self.soundVariStd_fear = features[3][3]

        self.soundlvlMean_happy = features[4][1]
        self.soundlvlStd_happy = features[5][1]
        self.soundlvlMean_sad = features[6][1]
        self.soundlvlStd_sad = features[7][1]
        self.soundlvlMean_angry = features[0][1]
        self.soundlvlStd_angry = features[1][1]
        self.soundlvlMean_fear = features[2][1]
        self.soundlvlStd_fear = features[3][1]

    # measurementsArray = [pitchlvl, pitchVari, soundVari, soundlvl]

    def getDistance(self, featureMeasurement, featureValue):
        distance = np.sqrt((featureValue - featureMeasurement) ** 2)

        return distance

    def getRelation(self, mean, std, measurementsArrayValue):
        relation = (mean + self.getDistance(measurementsArrayValue, mean)) / (mean + std)

        return relation

    def zeroDivision(self, n, d):
        return n / d if d > 0 else 100

    def checkPitch(self, measurementsArray):
        pHappy = self.getRelation(self.pitchMean_happy, self.pitchStd_happy, measurementsArray[0])
        pSad = self.getRelation(self.pitchMean_sad, self.pitchStd_sad, measurementsArray[0])
        pAngry = self.getRelation(self.pitchMean_angry, self.pitchStd_angry, measurementsArray[0])
        pFear = self.getRelation(self.pitchMean_fear, self.pitchStd_fear, measurementsArray[0])

        pitchRelations = [pHappy, pSad, pAngry, pFear]
        print('---------------------------')
        print("Pitch Happy:", pitchRelations[0])
        print("Pitch Sad:", pitchRelations[1])
        print("Pitch Angry:", pitchRelations[2])
        print("Pitch Fear:", pitchRelations[3])

        mostProbableMood = [min(pitchRelations), pitchRelations.index(min(pitchRelations))]
        return mostProbableMood

    def checkPitchVariance(self, measurementsArray):
        pvHappy = self.getRelation(self.pitchVariMean_happy, self.pitchVariStd_happy, measurementsArray[1])
        pvSad = self.getRelation(self.pitchVariMean_sad, self.pitchVariStd_sad, measurementsArray[1])
        pvAngry = self.getRelation(self.pitchVariMean_angry, self.pitchVariStd_angry, measurementsArray[1])
        pvFear = self.getRelation(self.pitchVariMean_sad, self.pitchVariStd_fear, measurementsArray[1])

        pitchVarianceRelations = [pvHappy, pvSad, pvAngry, pvFear]
        print('---------------------------')
        print("Pitch Variance Happy:", pitchVarianceRelations[0])
        print("Pitch Variance Sad:", pitchVarianceRelations[1])
        print("Pitch Variance Angry:", pitchVarianceRelations[2])
        print("Pitch Variance Fear:", pitchVarianceRelations[3])

        mostProbableMood = [min(pitchVarianceRelations), pitchVarianceRelations.index(min(pitchVarianceRelations))]
        return mostProbableMood

    def checkSoundVariance(self, measurementsArray):
        svHappy = self.getRelation(self.soundVariMean_happy, self.soundVariStd_happy, measurementsArray[2])
        svSad = self.getRelation(self.soundVariMean_sad, self.soundVariStd_sad, measurementsArray[2])
        svAngry = self.getRelation(self.soundVariMean_angry, self.soundVariStd_angry, measurementsArray[2])
        svFear = self.getRelation(self.soundVariMean_fear, self.soundVariStd_fear, measurementsArray[2])

        soundVarianceRelation = [svHappy, svSad, svAngry, svFear]
        print('---------------------------')
        print("Sound Variance Happy:", soundVarianceRelation[0])
        print("Sound Variance Sad:", soundVarianceRelation[1])
        print("Sound Variance Angry:", soundVarianceRelation[2])
        print("Sound Variance Fear:", soundVarianceRelation[3])

        mostProbableMood = [min(soundVarianceRelation), soundVarianceRelation.index(min(soundVarianceRelation))]
        return mostProbableMood

    def resetMoodScores(self):
        self.hScore, self.hValue = 0, 0
        self.sScore, self.sValue = 0, 0
        self.aScore, self.aValue = 0, 0
        self.fScore, self.fValue = 0, 0
        self.tScore, self.tValue = 0, 0

    def checkSound(self, measurementsArray):
        sHappy = self.getRelation(self.soundlvlMean_happy, self.soundlvlStd_happy, measurementsArray[3])
        sSad = self.getRelation(self.soundlvlMean_sad, self.soundlvlStd_sad, measurementsArray[3])
        sAngry = self.getRelation(self.soundlvlMean_angry, self.soundlvlStd_angry, measurementsArray[3])
        sFear = self.getRelation(self.soundlvlMean_fear, self.soundlvlStd_fear, measurementsArray[3])

        soundLevelRelation = [sHappy, sSad, sAngry, sFear]
        print('---------------------------')
        print("Sound Level Happy:", soundLevelRelation[0])
        print("Sound Level Sad:", soundLevelRelation[1])
        print("Sound Level Angry:", soundLevelRelation[2])
        print("Sound Level Fear:", soundLevelRelation[3])

        mostProbableMood = [min(soundLevelRelation), soundLevelRelation.index(min(soundLevelRelation))]
        return mostProbableMood

    def checkEmotion(self, measurementsArray):
        pitch = self.checkPitch(measurementsArray)
        pitchVariance = self.checkPitchVariance(measurementsArray)
        soundVariance = self.checkSoundVariance(measurementsArray)
        soundlvl = self.checkSound(measurementsArray)
        print('---------------------------')
        print("CheckPitch:", pitch)
        print("CheckPitchVariance:", pitchVariance)
        print("CheckSoundVariance:", soundVariance)
        print("CheckSound:", soundlvl)

        if pitch[1] == 0 and pitch[0] < 1:
            self.hValue += pitch[0]
            self.hScore += 1
            print("hScore awarded from pitch level")

        elif pitch[1] == 1 and pitch[0] < 1:
            self.sValue += pitch[0]
            self.sScore += 1
            print("sScore awarded from pitch level")

        elif pitch[1] == 2 and pitch[0] < 1:
            self.aValue += pitch[0]
            self.aScore += 1
            print("aScore awarded from pitch level")

        elif pitch[1] == 3 and pitch[0] < 1:
            self.fValue += pitch[0]
            self.fScore += 1
            print("fScore awarded from pitch level")

        else:
            print("Pitch not within any range")

        if pitchVariance[1] == 0 and pitchVariance[0] < 1:
            self.hValue += pitchVariance[0]
            self.hScore += 1
            print("hScore awarded from pitch variance")

        elif pitchVariance[1] == 1 and pitchVariance[0] < 1:
            self.sValue += pitchVariance[0]
            self.sScore += 1
            print("sScore awarded from pitch variance")

        elif pitchVariance[1] == 2 and pitchVariance[0] < 1:
            self.aValue += pitchVariance[0]
            self.aScore += 1
            print("aScore awarded from pitch variance")

        elif pitchVariance[1] == 3 and pitchVariance[0] < 1:
            self.fValue += pitchVariance[0]
            self.fScore += 1
            print("fScore awarded from pitch variance")

        else:
            print("Pitch variance not within any range")

        if soundVariance[1] == 0 and soundVariance[0] < 1:
            self.hValue += soundVariance[0]
            self.hScore += 1
            print("hScore awarded from sound variance")

        elif soundVariance[1] == 1 and soundVariance[0] < 1:
            self.sValue += soundVariance[0]
            self.sScore += 1
            print("sScore awarded from sound variance")

        elif soundVariance[1] == 2 and soundVariance[0] < 1:
            self.aValue += soundVariance[0]
            self.aScore += 1
            print("aScore awarded from sound variance")

        elif soundVariance[1] == 3 and soundVariance[0] < 1:
            self.fValue += soundVariance[0]
            self.fScore += 1
            print("fScore awarded from sound variance")

        else:
            print("Sound variance not within any range")

        if soundlvl[1] == 0 and soundlvl[0] < 1:
            self.hValue += soundlvl[0]
            self.hScore += 1
            print("hScore awarded from sound level")

        elif soundlvl[1] == 1 and soundlvl[0] < 1:
            self.sValue += soundlvl[0]
            self.sScore += 1
            print("sScore awarded from sound level")

        elif soundlvl[1] == 2 and soundlvl[0] < 1:
            self.aValue += soundlvl[0]
            self.aScore += 1
            print("aScore awarded from sound level")

        elif soundlvl[1] == 3 and soundlvl[0] < 1:
            self.fValue += soundlvl[0]
            self.fScore += 1
            print("fScore awarded from sound level")

        else:
            print("Sound level not within any range")

        theEmotionArray = [self.zeroDivision(self.hValue, self.hScore), self.zeroDivision(self.sValue, self.sScore),
                           self.zeroDivision(self.aValue, self.aScore), self.zeroDivision(self.fValue, self.fScore),
                           self.zeroDivision(self.tValue, self.tScore)]

        if self.hScore > 1:
            theEmotionArray[0] = theEmotionArray[0] * 0.70
            if self.hScore > 2:
                theEmotionArray[0] = theEmotionArray[0] * 0.70
                if self.hScore > 3:
                    theEmotionArray[0] = theEmotionArray[0] * 0.50

        if self.sScore > 1:
            theEmotionArray[1] = theEmotionArray[1] * 0.70
            if self.sScore > 2:
                theEmotionArray[1] = theEmotionArray[1] * 0.70
                if self.sScore > 3:
                    theEmotionArray[1] = theEmotionArray[1] * 0.50

        if self.aScore > 1:
            theEmotionArray[2] = theEmotionArray[2] * 0.70
            if self.aScore > 2:
                theEmotionArray[2] = theEmotionArray[2] * 0.70
                if self.aScore > 3:
                    theEmotionArray[2] = theEmotionArray[2] * 0.50

        if self.fScore > 1:
            theEmotionArray[3] = theEmotionArray[3] * 0.70
            if self.fScore > 2:
                theEmotionArray[3] = theEmotionArray[3] * 0.70
                if self.fScore > 3:
                    theEmotionArray[3] = theEmotionArray[3] * 0.50

        if self.tScore > 1:
            theEmotionArray[4] = theEmotionArray[4] * 0.70
            if self.tScore > 2:
                theEmotionArray[4] = theEmotionArray[4] * 0.70
                if self.tScore > 3:
                    theEmotionArray[4] = theEmotionArray[4] * 0.50

        print('---------------------------')
        print("hScore+:", self.hScore)
        print("sScore+:", self.sScore)
        print("aScore+:", self.aScore)
        print("fScore+:", self.fScore)
        print('---------------------------')
        print("Emotion value Happy:", theEmotionArray[0])
        print("Emotion value Sad:", theEmotionArray[1])
        print("Emotion value Angry:", theEmotionArray[2])
        print("Emotion value Fear:", theEmotionArray[3])
        print('---------------------------')

        if all(e > 1 for e in theEmotionArray):
            "No emotion detected, staying in same state"
            return self.prevState

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



        print("--------------------------------------------------------")
        print("---------------- THE END OF ANALYSIS -------------------")
        print("--------------------------------------------------------")
