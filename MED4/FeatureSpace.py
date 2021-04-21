import numpy as np
import math


class FeatureSpace:
    Emotions = {
        "Happy": happyMin and happyMax,
        "Sad": sadMin and sadMax,
        "Angry": angryMin and angryMax,
        "Fear": fearMin and fearMax
    }

    Features = {
        "pitchlvl": pitchMin and pitchMax,
        "pitchVari": pitchVariMin and pitchVariMax,
        "soundVari": soundVariMin and soundVariMax,
        "soundlvl": sondlvlMin and soundlvlMax,
    }
    hScore, hValue = 0, 0
    sScore, sValue = 0, 0
    aScore, aValue = 0, 0
    fScore, fValue = 0, 0
    tScore, tValue = 0, 0

    pitchMean_happy = 0
    pitchStd_happy = 0
    pitchMean_sad = 0
    pitchStd_sad = 0
    pitchMean_angry = 0
    pitchStd_angry = 0
    pitchMean_fear = 0
    pitchStd_fear = 0
    pitchMean_tender = 0
    pitchStd_tender = 0

    pitchVariMean_happy = 0
    pitchVariStd_happy = 0
    pitchVariMean_sad = 0
    pitchVariStd_sad = 0
    pitchVariMean_angry = 0
    pitchVariStd_angry = 0
    pitchVariMean_fear = 0
    pitchVariStd_fear = 0
    pitchVariMean_tender = 0
    pitchVariStd_tender = 0

    soundVariMean_happy = 0
    soundVariStd_happy = 0
    soundVariMean_sad = 0
    soundVariStd_sad = 0
    soundVariMean_angry = 0
    soundVariStd_angry = 0
    soundVariMean_fear = 0
    soundVariStd_fear = 0
    soundVariMean_tender = 0
    soundVariStd_tender = 0

    soundlvlMean_happy = 0
    soundlvlStd_happy = 0
    soundlvlMean_sad = 0
    soundlvlStd_sad = 0
    soundlvlMean_angry = 0
    soundlvlStd_angry = 0
    soundlvlMean_fear = 0
    soundlvlStd_fear = 0
    soundlvlMean_tender = 0
    soundlvlStd_tender = 0

    # pitchMin_happy = 0
    # pitchMax_happy = 0
    # pitchMin_sad = 0
    # pitchMax_sad = 0
    # pitchMin_angry = 0
    # pitchMax_angry = 0
    # pitchMin_fear = 0
    # pitchMax_fear = 0
    #
    # pitchVariMin_happy = 0
    # pitchVariMax_happy = 0
    # pitchVariMin_sad = 0
    # pitchVariMax_sad = 0
    # pitchVariMin_angry = 0
    # pitchVariMax_angry = 0
    # pitchVariMin_fear = 0
    # pitchVariMax_fear = 0
    #
    # soundVariMin_happy = 0
    # soundVariMax_happy = 0
    # soundVariMin_sad = 0
    # soundVariMax_sad = 0
    # soundVariMin_angry = 0
    # soundVariMax_angry = 0
    # soundVariMin_fear = 0
    # soundVariMax_fear = 0
    #
    # soundlvlMin_happy = 0
    # soundlvlMax_happy = 0
    # soundlvlMin_sad = 0
    # soundlvlMax_sad = 0
    # soundlvlMin_angry = 0
    # soundlvlMax_angry = 0
    # soundlvlMin_fear = 0
    # soundlvlMax_fear = 0

    # measurementsArray = [pitchlvl, pitchVari, soundVari, soundlvl]

    def getDistance(self, featureMeasurement, featureValue):
        distance = np.sqrt((featureValue - featureMeasurement) ** 2)

        return distance

    def getRelation(self, mean, std, measurementsArrayValue):
        relation = (mean + self.getDistance(measurementsArrayValue, mean)) / (mean + 2 * std)

        return relation

    def weirdDivision(self, n, d):
        return float(n / d) if d else 100

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
        print("Sound Variance Happy:", soundVarianceRelations[0])
        print("Sound Variance Sad:", soundVarianceRelations[1])
        print("Sound Variance Angry:", soundVarianceRelations[2])
        print("Sound Variance Fear:", soundVarianceRelations[3])
        print("Sound Variance Tender:", soundVarianceRelations[4])

        mostProbableMood = [min(soundVarianceRelation), soundVarianceRelation.index(min(soundVarianceRelation))]
        return mostProbableMood

    def checkSound(self, measurementsArray):
        sHappy = self.getRelation(self.soundlvlMean_happy, self.soundlvlStd_happy, measurementsArray[3])
        sSad = self.getRelation(self.soundlvlMean_sad, self.soundlvlStd_sad, measurementsArray[3])
        sAngry = self.getRelation(self.soundlvlMean_angry, self.soundlvlStd_angry, measurementsArray[3])
        sFear = self.getRelation(self.soundlvlMean_fear, self.soundlvlStd_fear, measurementsArray[3])
        sTender = self.getRelation(self.soundlvlMean_tender, self.soundlvlStd_tender, measurementsArray[3])

        soundLevelRelation = [sHappy, sSad, sAngry, sFear, sTender]
        print('---------------------------')
        print("Sound Level Happy:", soundVarianceRelations[0])
        print("Sound Level Sad:", soundVarianceRelations[1])
        print("Sound Level Angry:", soundVarianceRelations[2])
        print("Sound Level Fear:", soundVarianceRelations[3])
        print("Sound Level Tender:", soundVarianceRelations[4])

        mostProbableMood = [min(soundLevelRelation), soundLevelRelation.index(min(soundLevelRelation))]
        return mostProbableMood

    def checkEmotion(self, measurementsArray):
        pitch = self.checkPitch(measurementsArray)
        pitchVariance = self.checkPitchVariance(measurementsArray)
        soundVariance = self.checkSoundVariance(measurementsArray)
        soundlvl = self.checkSound(measurementsArray)

        if pitch[1] == 0:
            self.hValue + pitch[0], self.hScore + 1

        if pitch[1] == 1:
            self.sValue + pitch[0], self.sScore + 1

        if pitch[1] == 2:
            self.aValue + pitch[0], self.aScore + 1

        if pitch[1] == 3:
            self.fValue + pitch[0], self.fScore + 1

        if pitch[1] == 4:
            self.tValue + pitch[0], self.tScore + 1

        if pitchVariance[1] == 0:
            self.hValue + pitchVariance[0], self.hScore + 1

        if pitchVariance[1] == 1:
            self.sValue + pitchVariance[0], self.sScore + 1

        if pitchVariance[1] == 2:
            self.aValue + pitchVariance[0], self.aScore + 1

        if pitchVariance[1] == 3:
            self.fValue + pitchVariance[0], self.fScore + 1

        if pitchVariance[1] == 4:
            self.tValue + pitchVariance[0], self.tScore + 1

        if soundVariance[1] == 0:
            self.hValue + soundVariance[0], self.hScore + 1

        if soundVariance[1] == 1:
            self.sValue + soundVariance[0], self.sScore + 1

        if soundVariance[1] == 2:
            self.aValue + soundVariance[0], self.aScore + 1

        if soundVariance[1] == 3:
            self.fValue + soundVariance[0], self.fScore + 1

        if soundVariance[1] == 4:
            self.tValue + soundVariance[0], self.tScore + 1

        if soundlvl[1] == 0:
            self.hValue + soundlvl[0], self.hScore + 1

        if soundlvl[1] == 1:
            self.sValue + soundlvl[0], self.sScore + 1

        if soundlvl[1] == 2:
            self.aValue + soundlvl[0], self.aScore + 1

        if soundlvl[1] == 3:
            self.fValue + soundlvl[0], self.fScore + 1

        if soundlvl[1] == 4:
            self.tValue + soundlvl[0], self.tScore + 1

        theEmotionArray = [self.weirdDivision(self.hValue, self.hScore), self.weirdDivision(self.sValue, self.sScore),
                           self.weirdDivision(self.aValue, self.aScore), self.weirdDivision(self.fValue, self.fScore),
                           self.weirdDivision(self.tValue, self.tScore)]

        if theEmotionArray.index(min(theEmotionArray)) == 0:
            print("Happy")

        if theEmotionArray.index(min(theEmotionArray)) == 1:
            print("Sad")

        if theEmotionArray.index(min(theEmotionArray)) == 2:
            print("Angry")

        if theEmotionArray.index(min(theEmotionArray)) == 3:
            print("Fear")

        if theEmotionArray.index(min(theEmotionArray)) == 4:
            print("Tender")
