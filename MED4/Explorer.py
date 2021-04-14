class Explorer:
    # Paths
    audioFilePath = "Wav/"
    denoisedFilePath = "Wav/Denoised_Files/"
    graphFilePath = "Graphs/"
    filteredFilePath = "Wav/Filtered/"

    def getAudioFilePath(self):
        return self.audioFilePath

    def getGraphFilePath(self):
        return self.graphFilePath

    def getDenoisedFilePath(self):
        return self.denoisedFilePath

    def getFilteredFilePath(self):
        return self.filteredFilePath

    def wTxt(self, name, value: str, fileMode):
        txt = open(str(name) + ".txt", str(fileMode))
        txt.write(str(value))
        txt.close()
