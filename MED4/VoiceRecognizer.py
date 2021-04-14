class VoiceRecognizer:
    thresholdLow = 425.75
    thresholdHigh = 505.55

    def recognize(self, valueToDetect: float):
        if self.thresholdLow < valueToDetect < self.thresholdHigh:
            print("Recognized as: Human")
        else:
            print("Recognized as: Noise")
