class Explorer:
    # Paths
    audioFilePath = "Wav/"
    denoisedFilePath = "Wav/Denoised_Files/"
    graphFilePath = "Graphs/"

    def getAudioFilePath(self):
        return self.audioFilePath

    def getGraphFilePath(self):
        return self.graphFilePath

    def getDenoisedFilePath(self):
        return self.denoisedFilePath