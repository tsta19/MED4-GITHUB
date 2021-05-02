# import
from evaluation import Evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import numpy as np


def GetMeanAccuracy(iterations, emotions,methods):
    print(methods)
    for x in range(iterations):
        x_test, y_test = e.train(emotions, methods)

        checkList = []

        counter = 0
        for y in range(len(checkList)):

            if checkList[y] in y_test:
                counter += 1

        if counter > len(checkList)-1:
            conMatrixArr.append(e.test(x_test, y_test))

    print("------------------------------")
    print()

    print(len(conMatrixArr))

    accuracy = np.array([])

    for x in range(len(conMatrixArr)):
        accuracy = np.append(accuracy, accuracy_score(conMatrixArr[x][0], conMatrixArr[x][1]))
        print("Accuracy: " + str(accuracy[x]))
        print("Confusion Matrix:")
        print(confusion_matrix(conMatrixArr[x][0], conMatrixArr[x][1]))

        unique_numbers = list(set(conMatrixArr[x][0]))
        emotionStr = emotions[int(unique_numbers[0])]
        for i in range(1, len(unique_numbers)):
            emotionStr += " " + str(emotions[int(unique_numbers[i])])

        print(emotionStr)

        plt.show()
        print()

    return str(np.mean(accuracy))


if __name__ == '__main__':
    methods = [True, True, True, True, True]
    e = Evaluation()
    methodNames = ["Pitch", "Sound Level", "Pitch Variance", "Sound Level Variance", "Power Frequency"]
    emotions = ["Angry", "Fear", "Happiness", "Sad"]
    e.makeDataset(emotions)
    conMatrixArr = []
    iterations = 100

    acc = [[] for i in range(5)]
    accMethods = [[] for i in range(5)]
    print(len(acc))

    for i1 in range(len(methods)):
        methods = [False,False,False,False,False]
        methods[i1] = True
        acc[0].append(GetMeanAccuracy(iterations, emotions, methods))
        accMethods[0].append(methods)

        for i2 in range(i1+1,len(methods)):
            methods = [False, False, False, False, False]
            methods[i1],methods[i2] = True, True
            acc[1].append(GetMeanAccuracy(iterations, emotions, methods))
            accMethods[1].append(methods)

            for i3 in range(i1 + 2, len(methods)):
                methods = [False, False, False, False, False]
                methods[i1],methods[i2],methods[i3] = True, True, True
                acc[2].append(GetMeanAccuracy(iterations, emotions, methods))
                accMethods[2].append(methods)

                for i4 in range(i1 + 3, len(methods)):
                    methods = [False, False, False, False, False]
                    methods[i1],methods[i2],methods[i3],methods[i4] = True, True, True, True
                    acc[3].append(GetMeanAccuracy(iterations, emotions, methods))
                    accMethods[3].append(methods)

                    for i5 in range(i1 + 4, len(methods)):
                        methods = [True, True, True, True, True]
                        acc[4].append(GetMeanAccuracy(iterations, emotions, methods))
                        accMethods[4].append(methods)

    print("Mean Accuracy: " + str(acc))
    print("Methods used:" + str(accMethods))

    maxAcc = [max(acc[0]),max(acc[1]),max(acc[2]),max(acc[3]),max(acc[4])]

    print("Highest accuracy [0]: " + str(max(acc[0])))
    print("Highest accuracy [1]: " + str(max(acc[1])))
    print("Highest accuracy [2]: " + str(max(acc[2])))
    print("Highest accuracy [3]: " + str(max(acc[3])))
    print("Highest accuracy [4]: " + str(max(acc[4])))

    print("Highest overall accuracy: " + str(max(maxAcc)))


    index1 = maxAcc.index(max(maxAcc))
    index2 = acc[maxAcc.index(max(maxAcc))].index(max(maxAcc))
    print("Highest Overall accuracy found in acc[" + str(index1) + "][" + str(index2) + "]")

    methodStr = ""
    for x in range(len(accMethods[index1][index2])):
        if accMethods[index1][index2][x]:
            methodStr += " " + str(methodNames[x])
    print("The best combination of methods is:" + methodStr)

    #for x in range()

    """if __name__ == '__main__':

    # Modes for processing
    # Single File is a manual version where file name and outputName is specified
    # Directory is an automatic version that goes through files in a directory
    modeSingleFile = False
    modeDirectory = True
    debug = True
    record = False

    # Iterator(s)
    iteration = 1

    # Sample Frequency
    sampleFrequency = 44100

    # Arrays
    meanFrequencies = []

    # Class instantiations
    exp = Explorer()
    nRed = NoiseReduction()
    spectra = SpectraAnalysis()
    vr = VoiceRecognizer()

    if record:
        from Recorder import *

        rec = Recorder()
    else:
        print("Recording is turned: Off")

    if modeSingleFile:
        number = 1
        soundFile = "{}sound_file_{}.{}".format(exp.getAudioFilePath(), number, "wav")
        singleFileSpecification = os.path.join(str(soundFile))

        nRed.noiseFiltering(False, soundFile, singleFileSpecification)
        spectra.spectra("sound_file_1")
        sampleRate, data = wavfile.read(exp.getAudioFilePath() + "sound_file_1" + ".wav")
        vr.recognize(spectra.meanFrequency(data, sampleFrequency))

        if debug:
            print("Mean Frequency:", spectra.meanFrequency(data, sampleFrequency))

    if modeDirectory:
        soundFile = "mic"
        outputName = "filtered"
        for filename in glob.glob(os.path.join(exp.getAudioFilePath(), '*.wav')):
            wavFile = wave.open(filename, 'r')
            data = wavFile.readframes(wavFile.getnframes())
            wavFile.close()

        for audioFiles in glob.glob(os.path.join(exp.getAudioFilePath() + str(soundFile) + '*.wav')):
            audioFileSpecification = os.path.join(exp.getFilteredFilePath(), str(soundFile) + "_" + str(iteration) + "_"
                                                  + str(outputName) + ".wav")

            if debug:
                print("+------------------------+")
                print("For-Loop iteration:", iteration)
                print("File:", audioFileSpecification)

            nRed.noiseFiltering(True, soundFile, audioFileSpecification, audioFiles)
            spectra.spectra(audioFileSpecification)
            # sampleRate, data = wavfile.read(exp.getAudioFilePath() + soundFile + "_" + str(iteration) + "_"
            #                                 + str(outputName) + ".wav")
            samples, sampleRate = librosa.load((exp.getFilteredFilePath() + soundFile + "_" + str(iteration) + "_"
                                                + str(outputName) + ".wav"), sr=None, mono=True, offset=0.0,
                                               duration=None)

            if debug:
                print("Samples:", samples)
                print("SampleRate:", sampleRate)
                fileDuration = len(samples) / sampleRate
                print("File Duration:", round(fileDuration, 2), "second(s)")

            spectra.fastFourierTransform(samples, sampleRate)
            spectra.spectrogram(samples, sampleRate)

            # spectra.spectrogram(samples, sampleRate)

            meanFrequency = spectra.meanFrequency(samples, sampleFrequency)
            meanFrequencies.append(meanFrequency)

            if debug:
                print("Mean Frequency:", round(meanFrequency, 4))
                recognition = vr.recognize(meanFrequency)
                print("+------------------------+", "\n")

            iteration += 1
            spectra.iterator += 1"""

        # ----------------- WORK IN PROGRESS ---------------------------
        #     with open("Program_Summary_Data", "w") as txt:
        #         spacer = "+------------------------+"
        #         space = "\n"
        #         txt.write(spacer + space)
        #         txt.write("File: " + str(audioFileSpecification) + space)
        #         txt.write("Mean Frequencies: " + str(meanFrequencies) + space)
        #         txt.write("File Duration: " + str(len(samples)) + space)
        #         txt.write("Recognized as: " + str(recognition) + space)
        #         txt.write(spacer + space)
        #
        # txt.close()
        # ----------------- WORK IN PROGRESS ---------------------------

    # dm.soundDataManager(soundDirectory, graphDirectory, "sound_file_")
