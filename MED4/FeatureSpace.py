import numpy as np
import math


class FeatureSpace:
    hScore, hValue = 0, 0
    sScore, sValue = 0, 0
    aScore, aValue = 0, 0
    fScore, fValue = 0, 0
    tScore, tValue = 0, 0

    theEmotionScores = [hScore, sScore, aScore, fScore, tScore]

    pitchMean_happy = 107.36030064326853
    pitchStd_happy = 9.7060824240088
    pitchMean_sad = 102.55473052021894
    pitchStd_sad = 9.176241863297498
    pitchMean_angry = 119.4774692941384
    pitchStd_angry = 15.933613610064395
    pitchMean_fear = 132.34945603522874
    pitchStd_fear = 13.452992893809547
    pitchMean_tender = 113.60769985724662
    pitchStd_tender = 9.01256789027346

    pitchVariMean_happy = 57.21247841446778
    pitchVariStd_happy = 29.477671770697686
    pitchVariMean_sad = 37.9399161614971
    pitchVariStd_sad = 28.302112679388173
    pitchVariMean_angry = 67.89347660594153
    pitchVariStd_angry = 28.35744522689046
    pitchVariMean_fear = 90.83493760486154
    pitchVariStd_fear = 25.589842125148287
    pitchVariMean_tender = 65.73697819631049
    pitchVariStd_tender = 28.520669923988216

    soundVariMean_happy = 10.637224963253264
    soundVariStd_happy = 3.2093081907735157
    soundVariMean_sad = 6.869504579601349
    soundVariStd_sad = 4.1775828641675075
    soundVariMean_angry = 16.090760123707234
    soundVariStd_angry = 3.7181822634589965
    soundVariMean_fear = 12.594669178453046
    soundVariStd_fear = 4.819693699789634
    soundVariMean_tender = 9.27757095499356
    soundVariStd_tender = 4.377346340729348

    soundlvlMean_happy = 66.50103642574857
    soundlvlStd_happy = 1.4773186300113783
    soundlvlMean_sad = 62.519085453334675
    soundlvlStd_sad = 1.8658598150463226
    soundlvlMean_angry = 70.09925905153821
    soundlvlStd_angry = 3.039090517331451
    soundlvlMean_fear = 67.13781794326991
    soundlvlStd_fear = 1.6635825145896928
    soundlvlMean_tender = 66.82172943945206
    soundlvlStd_tender = 1.804376296123412

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

        if theEmotionArray.index(min(theEmotionArray)) == 1:
            print("Sad")

        if theEmotionArray.index(min(theEmotionArray)) == 2:
            print("Angry")

        if theEmotionArray.index(min(theEmotionArray)) == 3:
            print("Fear")

        if theEmotionArray.index(min(theEmotionArray)) == 4:
            print("Tender")

        self.hScore, self.hValue = 0, 0
        self.sScore, self.sValue = 0, 0
        self.aScore, self.aValue = 0, 0
        self.fScore, self.fValue = 0, 0
        self.tScore, self.tValue = 0, 0
        print("--------------------------------------------------------")
        print("---------------- THE END OF ANALYSIS -------------------")
        print("--------------------------------------------------------")
