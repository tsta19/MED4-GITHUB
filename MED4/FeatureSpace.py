import numpy as np
import math


class FeatureSpace:
    hScore, hValue = 0, 0
    sScore, sValue = 0, 0
    aScore, aValue = 0, 0
    fScore, fValue = 0, 0
    tScore, tValue = 0, 0

    theEmotionScores = [hScore, sScore, aScore, fScore, tScore]

    print("Getting imput from list")

    f = open("featurespacevariables.txt", "r")
    input = f.read()
    print(input)
    lines = input.replace("[","").replace("]","").strip().split('\n')
    print(len(lines))
    print(lines)
    features = [[0.0 for i in range(4)] for ii in range(len(lines)-2)]
    print(features)
    for i in range(2, len(lines)):
        arr = lines[i].split(", ")
        for x in range(len(arr)):
            features[i-2][x] = float(arr[x])

    print(features)
    print("Done getting input from list")

    pitchMean_happy = features[4][0]
    pitchStd_happy = features[5][0]
    pitchMean_sad = features[6][0]
    pitchStd_sad = features[7][0]
    pitchMean_angry = features[0][0]
    pitchStd_angry = features[1][0]
    pitchMean_fear = features[2][0]
    pitchStd_fear = features[3][0]
    pitchMean_tender = features[8][0]
    pitchStd_tender = features[9][0]

    pitchVariMean_happy = features[4][2]
    pitchVariStd_happy = features[5][2]
    pitchVariMean_sad = features[6][2]
    pitchVariStd_sad = features[7][2]
    pitchVariMean_angry = features[0][2]
    pitchVariStd_angry = features[1][2]
    pitchVariMean_fear = features[2][2]
    pitchVariStd_fear = features[3][2]
    pitchVariMean_tender = features[8][2]
    pitchVariStd_tender = features[9][2]

    soundVariMean_happy = features[4][3]
    soundVariStd_happy = features[5][3]
    soundVariMean_sad = features[6][3]
    soundVariStd_sad = features[7][3]
    soundVariMean_angry = features[0][3]
    soundVariStd_angry = features[1][3]
    soundVariMean_fear = features[2][3]
    soundVariStd_fear = features[3][3]
    soundVariMean_tender = features[8][3]
    soundVariStd_tender = features[9][3]

    soundlvlMean_happy = features[4][1]
    soundlvlStd_happy = features[5][1]
    soundlvlMean_sad = features[6][1]
    soundlvlStd_sad = features[7][1]
    soundlvlMean_angry = features[0][1]
    soundlvlStd_angry = features[1][1]
    soundlvlMean_fear = features[2][1]
    soundlvlStd_fear = features[3][1]
    soundlvlMean_tender = features[8][1]
    soundlvlStd_tender = features[9][1]

    # measurementsArray = [pitchlvl, soundlvl, pitchVari, soundVari]

    def getDistance(self, featureMeasurement, featureValue):
        distance = np.sqrt((featureValue - featureMeasurement) ** 2)

        return distance

    def getRelation(self, mean, std, measurementsArrayValue):
        relation = (mean + self.getDistance(measurementsArrayValue, mean)) / (mean + 2 * std)

        return relation

    def zeroDivision(self, n, d):
        return n / d if d > 0 else 100

    def checkPitch(self, measurementsArray):
        pHappy = self.getRelation(self.pitchMean_happy, self.pitchStd_happy, measurementsArray[0])
        pSad = self.getRelation(self.pitchMean_sad, self.pitchStd_sad, measurementsArray[0])
        pAngry = self.getRelation(self.pitchMean_angry, self.pitchStd_angry, measurementsArray[0])
        pFear = self.getRelation(self.pitchMean_fear, self.pitchStd_fear, measurementsArray[0])
        pTender = self.getRelation(self.pitchMean_tender, self.pitchStd_tender, measurementsArray[0])

        pitchRelations = [pHappy, pSad, pAngry, pFear, pTender]
        print('---------------------------')
        print("Pitch Happy:", pitchRelations[0])
        print("Pitch Sad:", pitchRelations[1])
        print("Pitch Angry:", pitchRelations[2])
        print("Pitch Fear:", pitchRelations[3])
        print("Pitch Tender:", pitchRelations[4])

        mostProbableMood = [min(pitchRelations), pitchRelations.index(min(pitchRelations))]
        return mostProbableMood

    def checkPitchVariance(self, measurementsArray):
        pvHappy = self.getRelation(self.pitchVariMean_happy, self.pitchVariStd_happy, measurementsArray[1])
        pvSad = self.getRelation(self.pitchVariMean_sad, self.pitchVariStd_sad, measurementsArray[1])
        pvAngry = self.getRelation(self.pitchVariMean_angry, self.pitchVariStd_angry, measurementsArray[1])
        pvFear = self.getRelation(self.pitchVariMean_sad, self.pitchVariStd_fear, measurementsArray[1])
        pvTender = self.getRelation(self.pitchVariMean_tender, self.pitchVariStd_tender, measurementsArray[1])

        pitchVarianceRelations = [pvHappy, pvSad, pvAngry, pvFear, pvTender]
        print('---------------------------')
        print("Pitch Variance Happy:", pitchVarianceRelations[0])
        print("Pitch Variance Sad:", pitchVarianceRelations[1])
        print("Pitch Variance Angry:", pitchVarianceRelations[2])
        print("Pitch Variance Fear:", pitchVarianceRelations[3])
        print("Pitch Variance Tender:", pitchVarianceRelations[4])

        mostProbableMood = [min(pitchVarianceRelations), pitchVarianceRelations.index(min(pitchVarianceRelations))]
        return mostProbableMood

    def checkSoundVariance(self, measurementsArray):
        svHappy = self.getRelation(self.soundVariMean_happy, self.soundVariStd_happy, measurementsArray[2])
        svSad = self.getRelation(self.soundVariMean_sad, self.soundVariStd_sad, measurementsArray[2])
        svAngry = self.getRelation(self.soundVariMean_angry, self.soundVariStd_angry, measurementsArray[2])
        svFear = self.getRelation(self.soundVariMean_fear, self.soundVariStd_fear, measurementsArray[2])
        svTender = self.getRelation(self.soundVariMean_tender, self.soundVariStd_tender, measurementsArray[2])

        soundVarianceRelation = [svHappy, svSad, svAngry, svFear, svTender]
        print('---------------------------')
        print("Sound Variance Happy:", soundVarianceRelation[0])
        print("Sound Variance Sad:", soundVarianceRelation[1])
        print("Sound Variance Angry:", soundVarianceRelation[2])
        print("Sound Variance Fear:", soundVarianceRelation[3])
        print("Sound Variance Tender:", soundVarianceRelation[4])

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
        sTender = self.getRelation(self.soundlvlMean_tender, self.soundlvlStd_tender, measurementsArray[3])

        soundLevelRelation = [sHappy, sSad, sAngry, sFear, sTender]
        print('---------------------------')
        print("Sound Level Happy:", soundLevelRelation[0])
        print("Sound Level Sad:", soundLevelRelation[1])
        print("Sound Level Angry:", soundLevelRelation[2])
        print("Sound Level Fear:", soundLevelRelation[3])
        print("Sound Level Tender:", soundLevelRelation[4])

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

        if pitch[1] == 0:
            self.hValue += pitch[0]
            self.hScore += 1
            print("hScore awarded from pitch level")

        elif pitch[1] == 1:
            self.sValue += pitch[0]
            self.sScore += 1
            print("sScore awarded from pitch level")

        elif pitch[1] == 2:
            self.aValue += pitch[0]
            self.aScore += 1
            print("aScore awarded from pitch level")

        elif pitch[1] == 3:
            self.fValue += pitch[0]
            self.fScore += 1
            print("fScore awarded from pitch level")

        elif pitch[1] == 4:
            self.tValue += pitch[0]
            self.tScore += 1
            print("tScore awarded from pitch level")

        if pitchVariance[1] == 0:
            self.hValue += pitchVariance[0]
            self.hScore += 1
            print("hScore awarded from pitch variance")

        elif pitchVariance[1] == 1:
            self.sValue += pitchVariance[0]
            self.sScore += 1
            print("sScore awarded from pitch variance")

        elif pitchVariance[1] == 2:
            self.aValue += pitchVariance[0]
            self.aScore += 1
            print("aScore awarded from pitch variance")

        elif pitchVariance[1] == 3:
            self.fValue += pitchVariance[0]
            self.fScore += 1
            print("fScore awarded from pitch variance")

        elif pitchVariance[1] == 4:
            self.tValue += pitchVariance[0]
            self.tScore += 1
            print("tScore awarded from pitch variance")

        if soundVariance[1] == 0:
            self.hValue += soundVariance[0]
            self.hScore += 1
            print("hScore awarded from sound variance")

        elif soundVariance[1] == 1:
            self.sValue += soundVariance[0]
            self.sScore += 1
            print("sScore awarded from sound variance")

        elif soundVariance[1] == 2:
            self.aValue += soundVariance[0]
            self.aScore += 1
            print("aScore awarded from sound variance")

        elif soundVariance[1] == 3:
            self.fValue += soundVariance[0]
            self.fScore += 1
            print("fScore awarded from sound variance")

        elif soundVariance[1] == 4:
            self.tValue += soundVariance[0]
            self.tScore += 1
            print("tScore awarded from sound variance")

        if soundlvl[1] == 0:
            self.hValue += soundlvl[0]
            self.hScore += 1
            print("hScore awarded from sound level")


        elif soundlvl[1] == 1:
            self.sValue += soundlvl[0]
            self.sScore += 1
            print("sScore awarded from sound level")


        elif soundlvl[1] == 2:
            self.aValue += soundlvl[0]
            self.aScore += 1
            print("aScore awarded from sound level")


        elif soundlvl[1] == 3:
            self.fValue += soundlvl[0]
            self.fScore += 1
            print("fScore awarded from sound level")


        elif soundlvl[1] == 4:
            self.tValue += soundlvl[0]
            self.tScore += 1
            print("tScore awarded from sound level")

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
        print("tScore+:", self.tScore)
        print('---------------------------')
        print("Emotion value Happy:", theEmotionArray[0])
        print("Emotion value Sad:", theEmotionArray[1])
        print("Emotion value Angry:", theEmotionArray[2])
        print("Emotion value Fear:", theEmotionArray[3])
        print("Emotion value Tender:", theEmotionArray[4])
        print('---------------------------')
        if theEmotionArray.index(min(theEmotionArray)) == 0:
            print("Happy")
            self.resetMoodScores()
            return 0

        if theEmotionArray.index(min(theEmotionArray)) == 1:
            print("Sad")
            self.resetMoodScores()
            return 1

        if theEmotionArray.index(min(theEmotionArray)) == 2:
            print("Angry")
            self.resetMoodScores()
            return 2

        if theEmotionArray.index(min(theEmotionArray)) == 3:
            print("Fear")
            self.resetMoodScores()
            return 3

        if theEmotionArray.index(min(theEmotionArray)) == 4:
            print("Tender")
            self.resetMoodScores()
            return 4

        print("--------------------------------------------------------")
        print("---------------- THE END OF ANALYSIS -------------------")
        print("--------------------------------------------------------")

