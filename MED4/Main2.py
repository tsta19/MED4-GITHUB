# import
from evaluation import Evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np


def GetMeanAccuracy(iterations, emotions, methods):
    print(methods)
    accuracy = np.array([])
    x = np.array([])
    y = np.array([])


    for i in range(iterations):
        x_test, y_test = e.train(emotions, methods)

        conMatrix = e.test(x_test, y_test)

        accuracy = np.append(accuracy, accuracy_score(conMatrix[0], conMatrix[1]))
        x = np.append(x,  conMatrix[0])
        y = np.append(y, conMatrix[1])
        print("Accuracy: " + str(accuracy[i]))
        print("Confusion Matrix:")
        print(confusion_matrix(conMatrix[0], conMatrix[1]))

        """unique_numbers = list(set(conMatrix[x][0]))
        emotionStr = emotions[int(unique_numbers[0])]
        for i in range(1, len(unique_numbers)):
            emotionStr += " " + str(emotions[int(unique_numbers[i])])
            
        print(emotionStr)"""
        print()

    """indexArr = emotions
    indexArr.append("None")

    df_cm = pd.DataFrame(confusion_matrix(x, y, normalize='true'), index=[e for e in indexArr], columns=[e for e in indexArr])

    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=plt.cm.Blues)  # font size
    plt.yticks( va='center')

    print("Confusion Matrix:")
    print(confusion_matrix(x, y))
    print("Accuracy: " + str(accuracy_score(x, y)))

    cm = confusion_matrix(x,y)

    cm = np.delete(cm,len(cm)-1,0)
    print(cm)
    plt.show()"""

    return x, y


def evalBestMethod(iterations, emotions, methods):
    acc = [[] for i in range(4)]
    accMethods = [[] for i in range(4)]

    xArr = [[] for i in range(4)]
    yArr = [[] for i in range(4)]

    for i1 in range(len(methods)-1):
        methods = [False, False, False, False, False]
        methods[i1] = True
        x, y = GetMeanAccuracy(iterations, emotions, methods)
        xArr[0].append(x)
        yArr[0].append(y)
        acc[0].append(accuracy_score(x, y))
        accMethods[0].append(methods)

        for i2 in range(i1 + 1, len(methods)-1):
            methods = [False, False, False, False, False]
            methods[i1], methods[i2] = True, True
            x, y = GetMeanAccuracy(iterations, emotions, methods)
            xArr[1].append(x)
            yArr[1].append(y)
            acc[1].append(accuracy_score(x, y))
            accMethods[1].append(methods)

            for i3 in range(i2 + 1, len(methods)-1):
                methods = [False, False, False, False, False]
                methods[i1], methods[i2], methods[i3] = True, True, True
                x, y = GetMeanAccuracy(iterations, emotions, methods)
                xArr[2].append(x)
                yArr[2].append(y)
                acc[2].append(accuracy_score(x, y))
                accMethods[2].append(methods)

    methods = [True, True, True, True, False]
    x, y = GetMeanAccuracy(iterations, emotions, methods)
    xArr[3].append(x)
    yArr[3].append(y)
    acc[3].append(accuracy_score(x, y))
    accMethods[3].append(methods)

    for i in range(len(accMethods)):
        print(len(accMethods[i]))

    for i in range(len(xArr)):
        for ii in range(len(xArr[i])):
            print("Confusion Matrix for method-pair " + str(i))
            print(confusion_matrix(xArr[i][ii], yArr[i][ii]))

    acc = [[float(x) for x in i] for i in acc]

    print(f"Mean Accuracy: {acc}")
    print("Methods used:" + str(accMethods))

    maxAcc = [float(max(acc[0])), float(max(acc[1])), float(max(acc[2])), float(max(acc[3]))]

    print("Highest accuracy [0]: " + str(maxAcc[0]))
    print("Highest accuracy [1]: " + str(maxAcc[1]))
    print("Highest accuracy [2]: " + str(maxAcc[2]))
    print("Highest accuracy [3]: " + str(maxAcc[3]))

    print(f"Highest overall accuracy: {max(maxAcc)}")

    index1 = maxAcc.index(max(maxAcc))
    index2 = acc[index1].index(max(maxAcc))
    print("Highest Overall accuracy found in acc[" + str(index1) + "][" + str(index2) + "]")

    methodStr = ""
    for x in range(len(accMethods[index1][index2])):
        if accMethods[index1][index2][x]:
            methodStr += " " + str(methodNames[x])
    print("The best combination of methods is:" + methodStr)


def evalBestVoiceRange(iterations, emotions, methods, noiseRange, voiceRange, chunkRange):

    acc = []
    xArr = []
    yArr = []
    for x in range(voiceRange[0], voiceRange[1]):
        e.makeDatasetFromSound(emotions, noiseRange, x, chunkRange)

        x,y = GetMeanAccuracy(iterations, emotions, methods)
        xArr.append(x)
        yArr.append(y)

        acc.append(accuracy_score(x, y))

    for i in range(len(xArr)):
        print("Confusion Matrix for chunk size " + str(i))
        print(confusion_matrix(xArr[i], yArr[i]))

    print(acc)

    print("The Best chunk size is ")
    print("The Accuracy at that chunk size was " + str(max(acc)))

def evalBestNoiseRange(iterations, emotions, methods, noiseRange, voiceRange, chunkRange):

    acc = []
    xArr = []
    yArr = []
    for x in range(noiseRange[0], noiseRange[1]):
        e.makeDatasetFromSound(emotions, x, voiceRange, chunkRange)

        x,y = GetMeanAccuracy(iterations, emotions, methods)
        xArr.append(x)
        yArr.append(y)

        acc.append(accuracy_score(x, y))

    for i in range(len(xArr)):
        print("Confusion Matrix for chunk size " + str(i))
        print(confusion_matrix(xArr[i], yArr[i]))

    print(acc)

    print("The Best chunk size is ")
    print("The Accuracy at that chunk size was " + str(max(acc)))


def evalBestChunkSize(iterations, emotions, methods, noiseRange, voiceRange, chunkRange):

    acc = []
    xArr = []
    yArr = []
    for x in range(chunkRange[0], chunkRange[1], 100):
        e.makeDatasetFromSound(emotions, noiseRange, voiceRange, x)

        x,y = GetMeanAccuracy(iterations, emotions, methods)
        xArr.append(x)
        yArr.append(y)

        acc.append(accuracy_score(x, y))

    for i in range(len(xArr)):
        print("Confusion Matrix for chunk size " + str(chunkRange[0]*i))
        print(confusion_matrix(xArr[i], yArr[i]))

    print(acc)

    print("The Best chunk size is " + str((acc.index(max(acc)) * 100)))
    print("The Accuracy at that chunk size was " + str(max(acc)))


if __name__ == '__main__':
    methods = [True, True, True, True, False]
    e = Evaluation()
    methodNames = ["Pitch", "Sound Level", "Pitch Variance", "Sound Level Variance", "Power Frequency"]
    emotions = ["Angry", "Fear", "Happy", "Sad"]
    dataSetDone = False
    evalBest = True
    noiseRange = [3, 6]
    voiceRange = [1, 15]
    chunkRange = [2600, 3000]
    iterations = 100

    #arr =  [[1388,  535,  806,  102,    0],[ 424, 1620,    5,   17,  441],[ 484,  120, 1410,  465,   25],[   0,    0,  528, 2118,  412],[   0,    0,    0,    0,    0]]

    #y = [1,2,0,2]
    #x = [1,0,0,2]
    #print(confusion_matrix(x,y, normalize='true'))

    # evalBestNoiseRange(iterations, emotions, methods, 10)
    if evalBest is False:
        if dataSetDone:
            e.makeDatasetFromText(emotions)
        else:
            e.makeDatasetFromSound(emotions, noiseRange[0], voiceRange[0], chunkRange[0])
    else:
        evalBestVoiceRange(iterations, emotions, methods, noiseRange[0], voiceRange, chunkRange[0])
        # evalBestNoiseRange(iterations, emotions, methods,noiseRange, voiceRange[0], chunkRange[0])
        # evalBestMethod(iterations, emotions, methods)
        # evalBestChunkSize(iterations, emotions, methods, noiseRange[0], voiceRange[0], chunkRange)
    #evalBestMethod(iterations, emotions, methods)
    # print("Accuracy: " + str(GetMeanAccuracy(iterations, emotions, methods)))

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
