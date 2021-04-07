class VoiceRecognizer:
    thresholdLow = 49.2
    thresholdHigh = 59.4

    def recognize(self, valueToDetect: float):
        if self.thresholdLow < valueToDetect < self.thresholdHigh:
            print("Recognized as: Human")
        else:
            print("Recognized as: Noise")
