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

    #measurementsArray = [pitchlvl, pitchVari, soundVari, soundlvl]

    def getDistance(self, featureMeasurement, featureValue):
        distance = np.sqrt((featureValue - featureMeasurement)**2)

        return distance

    def getRelation(self, mean, std, measurementsArrayValue):
        relation = (mean + self.getDistance(measurementsArrayValue, mean)) / (mean + 2*std)

        return relation

    def checkPitch(self, measurementsArray):
        pHappy = self.getRelation(self.pitchMean_happy, self.pitchStd_happy, measurementsArray[0])
        pSad = self.getRelation(self.pitchMean_sad, self.pitchStd_sad, measurementsArray[0])
        pAngry = self.getRelation(self.pitchMean_angry, self.pitchStd_angry, measurementsArray[0])
        pFear = self.getRelation(self.pitchMean_fear, self.pitchStd_fear, measurementsArray[0])
        pTender = self.getRelation(self.pitchMean_tender, self.pitchStd_tender, measurementsArray[0])

        pitchRelations = [pHappy, pSad, pAngry, pFear, pTender]
        print('---------------------------')
        print("Pitch Happy%:", pitchRelations[0]*100)
        print("Pitch Sad%:", pitchRelations[1]*100)
        print("Pitch Angry%:", pitchRelations[2]*100)
        print("Pitch Fear%:", pitchRelations[3]*100)
        print("Pitch Tender%:", pitchRelations[4]*100)
        return pitchRelations

    def checkPitchVariance(self, measurementsArray):
        pvHappy = self.getRelation(self.pitchVariMean_happy, self.pitchVariStd_happy, measurementsArray[1])
        pvSad = self.getRelation(self.pitchVariMean_sad, self.pitchVariStd_sad, measurementsArray[1])
        pvAngry = self.getRelation(self.pitchVariMean_angry, self.pitchVariStd_angry, measurementsArray[1])
        pvFear = self.getRelation(self.pitchVariMean_sad, self.pitchVariStd_fear, measurementsArray[1])
        pvTender = self.getRelation(self.pitchVariMean_tender, self.pitchVariStd_tender, measurementsArray[1])

        pitchVarianceRelations = [pvHappy, pvSad, pvAngry, pvFear, pvTender]
        print('---------------------------')
        print("Pitch Variance Happy%:", pitchVarianceRelations[0] * 100)
        print("Pitch Variance Sad%:", pitchVarianceRelations[1] * 100)
        print("Pitch Variance Angry%:", pitchVarianceRelations[2] * 100)
        print("Pitch Variance Fear%:", pitchVarianceRelations[3] * 100)
        print("Pitch Variance Tender%:", pitchVarianceRelations[4] * 100)
        return pitchVarianceRelations

    def checkSoundVariance(self, measurementsArray):
        svHappy = self.getRelation(self.soundVariMean_happy, self.soundVariStd_happy, measurementsArray[2])
        svSad = self.getRelation(self.soundVariMean_sad, self.soundVariStd_sad, measurementsArray[2])
        svAngry = self.getRelation(self.soundVariMean_angry, self.soundVariStd_angry, measurementsArray[2])
        svFear = self.getRelation(self.soundVariMean_fear, self.soundVariStd_fear, measurementsArray[2])
        svTender = self.getRelation(self.soundVariMean_tender, self.soundVariStd_tender, measurementsArray[2])

        soundVarianceRelation = [svHappy, svSad, svAngry, svFear, svTender]
        print('---------------------------')
        print("Sound Variance Happy%:", soundVarianceRelations[0] * 100)
        print("Sound Variance Sad%:", soundVarianceRelations[1] * 100)
        print("Sound Variance Angry%:", soundVarianceRelations[2] * 100)
        print("Sound Variance Fear%:", soundVarianceRelations[3] * 100)
        print("Sound Variance Tender%:", soundVarianceRelations[4] * 100)
        return soundVarianceRelation


    def checkSound(self, measurementsArray):

        pass

    def checkEmotion(self):

        pass