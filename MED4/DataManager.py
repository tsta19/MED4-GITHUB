import math
import os
import sys
import pitch
from tqdm import tqdm
import time
import numpy as np
import wave
import matplotlib.pyplot as plt
import warnings



class DataManager:
    # Toggles
    showFullArray = False

    # Directory Settings
    testDirectory = ""
    savePath = ""
    processedDirectory = "Sound_Files/Sorted_Mic_Sample_Values_Processed"
    soundDirectory = "Sound_Files/Mic_Samples"
    soundGraphDirectory = "Sound_Files/Sorted_Mic_Sample_Graphs"

    # Variables
    executionTime = time.time()
    iteration = 0
    soundIteration = 1
    resultIteration = 1
    txtFileType = ".txt"
    soundFileType = ".wav"
    specifySoundFile = "mic_sample_"
    # p = pitch.find_pitch("Sound_Files/Mic_Samples/mic_sample_1.wav")
    # print(p)

    if showFullArray:
        np.set_printoptions(threshold=sys.maxsize)

    # Sound Related
    sampleRate = 44100
    warnings.simplefilter("ignore", DeprecationWarning)

    def writeTextFile(self, name, value):
        fileName = os.path.join(self.savePath, name + ".txt")
        txt = open(fileName, "w+")
        txt.write(value)

    def dataManager(self):
        dataMangerCode = "Function: dataManager"
        userInput = input("Image Directory:" + "\n")
        if userInput == "smsv":
            self.testDirectory = "Sorted_Mic_Sample_Values"
        else:
            self.testDirectory = userInput
        userInput = input("Save Path:" + "\n")
        if userInput == "smsvp":
            self.savePath = "Sorted_Mic_Sample_Values_Processed"
        else:
            self.savePath = userInput
        print("+-------------------------+")
        print("| Data Processing Started.")
        print("| Directory:", self.testDirectory, )
        print("| Save Path:", self.savePath)
        print("+-------------------------+")
        print("| Data in Directory:", len(os.listdir(self.testDirectory)))
        print("+-------------------------+")
        for data in tqdm(os.listdir(self.testDirectory)):
            if data.endswith(".txt"):
                print(data)
                self.iteration += 1
                self.writeTextFile("Processed_" + str(self.iteration), "Processed")
            else:
                print("No DATA was found", "ERROR: Wrong filetype")
        print(dataMangerCode, "operations are Done")

    def printTestingResults(self):
        printResultsCode = "Function: printTestingResults"
        print("+-------- Test Results --------+")
        print("Images tested:", self.iteration)
        print("Execution time: %s" % round((time.time() - self.executionTime), 2), "seconds")
        print("+-------- DATA EVALUATION ---------+")

        for data in tqdm(os.listdir(self.processedDirectory)):

            processedFiles = os.path.join(self.processedDirectory, "Processed_" + str(self.resultIteration) + ".txt")
            print("********* " + "Processed_" + str(self.resultIteration) + ".txt" + " *********")
            if data.endswith(".txt"):
                with open(processedFiles, "r") as dataFile:
                    print(dataFile.readlines())
            else:
                print("No DATA was found", "ERROR: Wrong filetype")
            self.resultIteration += 1

        print("+------------------------------+")
        print(printResultsCode, "operations are Done")

    def soundDataManager(self, soundDirectory, graphDirectory, fileName):
        soundFileCode = "Function: soundDataManager"
        for soundFiles in tqdm(os.listdir(soundDirectory)):
            soundFilesSpec = os.path.join(soundDirectory, str(fileName) + str(self.soundIteration) + ".wav")
            if soundFiles.endswith(".wav"):
                soundFile = wave.open(soundFilesSpec, "r")
                da = np.fromstring(soundFile.readframes(self.sampleRate), dtype=np.int16)
                left, right = da[0::2], da[1::2]
                lf, rf = abs(np.fft.rfft(left)), abs(np.fft.rfft(right))
                plt.figure(self.soundIteration)
                a = plt.subplot(211)
                r = 2 ** 16 / 2
                a.set_ylim([-r, r])
                a.set_xlabel('Time in Seconds   ' + str(fileName) + str(self.soundIteration))
                a.set_ylabel('Sample Value [-]')
                x = np.arange(22050) / 22050
                plt.plot(x, left)
                b = plt.subplot(212)
                b.set_xscale('log')
                b.set_xlabel('Frequency [Hz]   ' + str(fileName) + str(self.soundIteration))
                b.set_ylabel('|Amplitude|')
                plt.plot(lf)
                plt.tight_layout()
                plt.savefig(str(graphDirectory) + "/" + str(fileName) + str(self.soundIteration) + '.png')
                self.soundIteration += 1
            else:
                print("ERROR: Wrong filetype" + "\n", "It is supposed to be ", self.soundFileType)
        print(soundFileCode, "operations are Done")