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

def evalBestEmotion(iterations, emotions, methods):
    acc = [[] for i in range(5)]
    accEmotions = [[] for i in range(5)]

    xArr = [[] for i in range(5)]
    yArr = [[] for i in range(5)]

    ogEmotions = emotions

    for i1 in range(len(methods)):
        emotions = [ogEmotions[i1]]
        print(emotions)
        x, y = GetMeanAccuracy(iterations, emotions, methods)
        xArr[0].append(x)
        yArr[0].append(y)
        acc[0].append(accuracy_score(x, y))
        accEmotions[0].append(methods)

        for i2 in range(i1 + 1, len(methods)):
            emotions = [ogEmotions[i1], ogEmotions[i2]]
            print(emotions)
            x, y = GetMeanAccuracy(iterations, emotions, methods)
            xArr[1].append(x)
            yArr[1].append(y)
            acc[1].append(accuracy_score(x, y))
            accEmotions[1].append(methods)

            for i3 in range(i2 + 1, len(methods)):
                emotions = [ogEmotions[i1], ogEmotions[i2], ogEmotions[i3]]
                x, y = GetMeanAccuracy(iterations, emotions, methods)
                xArr[2].append(x)
                yArr[2].append(y)
                acc[2].append(accuracy_score(x, y))
                accEmotions[2].append(methods)

    emotions = ogEmotions
    x, y = GetMeanAccuracy(iterations, emotions, methods)
    xArr[3].append(x)
    yArr[3].append(y)
    acc[3].append(accuracy_score(x, y))
    accEmotions[3].append(methods)

    for i in range(len(accEmotions)):
        print(len(accEmotions[i]))

    for i in range(len(xArr)):
        for ii in range(len(xArr[i])):
            print("Confusion Matrix for emotion-pair " + str(i))
            print(confusion_matrix(xArr[i][ii], yArr[i][ii]))

    acc = [[float(x) for x in i] for i in acc]

    print(f"Mean Accuracy: {acc}")
    print("Emotions used:" + str(accEmotions))

    maxAcc = [float(max(acc[0])), float(max(acc[1])), float(max(acc[2])), float(max(acc[3])), float(max(acc[4]))]

    print("Highest accuracy [0]: " + str(maxAcc[0]))
    print("Highest accuracy [1]: " + str(maxAcc[1]))
    print("Highest accuracy [2]: " + str(maxAcc[2]))
    print("Highest accuracy [3]: " + str(maxAcc[3]))
    print("Highest accuracy [4]: " + str(maxAcc[4]))

    print(f"Highest overall accuracy: {max(maxAcc)}")

    index1 = maxAcc.index(max(maxAcc))
    index2 = acc[index1].index(max(maxAcc))
    print("Highest Overall accuracy found in acc[" + str(index1) + "][" + str(index2) + "]")



def evalBestMethod(iterations, emotions, methods):
    acc = [[] for i in range(5)]
    accMethods = [[] for i in range(5)]

    xArr = [[] for i in range(5)]
    yArr = [[] for i in range(5)]

    for i1 in range(len(methods)):
        methods = [False, False, False, False, False]
        methods[i1] = True
        x, y = GetMeanAccuracy(iterations, emotions, methods)
        xArr[0].append(x)
        yArr[0].append(y)
        acc[0].append(accuracy_score(x, y))
        accMethods[0].append(methods)

        for i2 in range(i1 + 1, len(methods)):
            methods = [False, False, False, False, False]
            methods[i1], methods[i2] = True, True
            x, y = GetMeanAccuracy(iterations, emotions, methods)
            xArr[1].append(x)
            yArr[1].append(y)
            acc[1].append(accuracy_score(x, y))
            accMethods[1].append(methods)

            for i3 in range(i2 + 1, len(methods)):
                methods = [False, False, False, False, False]
                methods[i1], methods[i2], methods[i3] = True, True, True
                x, y = GetMeanAccuracy(iterations, emotions, methods)
                xArr[2].append(x)
                yArr[2].append(y)
                acc[2].append(accuracy_score(x, y))
                accMethods[2].append(methods)

                for i4 in range(i3 + 1, len(methods)):
                    methods = [False, False, False, False, False]
                    methods[i1], methods[i2], methods[i3], methods[i4] = True, True, True, True
                    x, y = GetMeanAccuracy(iterations, emotions, methods)
                    xArr[3].append(x)
                    yArr[3].append(y)
                    acc[3].append(accuracy_score(x, y))
                    accMethods[3].append(methods)

    methods = [True, True, True, True, True]
    x, y = GetMeanAccuracy(iterations, emotions, methods)
    xArr[4].append(x)
    yArr[4].append(y)
    acc[4].append(accuracy_score(x, y))
    accMethods[4].append(methods)

    for i in range(len(accMethods)):
        print(len(accMethods[i]))

    for i in range(len(xArr)):
        for ii in range(len(xArr[i])):
            print("Confusion Matrix for method-pair " + str(i))
            print(confusion_matrix(xArr[i][ii], yArr[i][ii]))

    acc = [[float(x) for x in i] for i in acc]

    print(f"Mean Accuracy: {acc}")
    print("Methods used:" + str(accMethods))

    maxAcc = [float(max(acc[0])), float(max(acc[1])), float(max(acc[2])), float(max(acc[3])), float(max(acc[4]))]

    print("Highest accuracy [0]: " + str(maxAcc[0]))
    print("Highest accuracy [1]: " + str(maxAcc[1]))
    print("Highest accuracy [2]: " + str(maxAcc[2]))
    print("Highest accuracy [3]: " + str(maxAcc[3]))
    print("Highest accuracy [4]: " + str(maxAcc[4]))

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
    for x in range(chunkRange[0], chunkRange[1], 1):
        e.makeDatasetFromSound(emotions, noiseRange, voiceRange, 2** x)

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
    methods = [False, True, False, True, False]
    e = Evaluation()
    methodNames = ["Pitch", "Sound Level", "Pitch Variance", "Sound Level Variance", "Power Frequency"]
    emotions = ["Happy", "Angry", "Fear"]
    dataSetDone = True
    evalBest = False
    noiseRange = [3, 6]
    voiceRange = [1, 6]
    chunkRange = [1024, 13]
    iterations = 100

    #arr =  [[1388,  535,  806,  102,    0],[ 424, 1620,    5,   17,  441],[ 484,  120, 1410,  465,   25],[   0,    0,  528, 2118,  412],[   0,    0,    0,    0,    0]]

    #y = [1,2,0,2]
    #x = [1,0,0,2]
    #print(confusion_matrix(x,y, normalize='true'))

    # evalBestNoiseRange(iterations, emotions, methods, 10)
    #arr =  [[0.6307407407407407, 0.0, 0.0, 0.0, 0.16925925925925925], [0.6129629629629629, 0.6240740740740741, 0.6208641975308642, 0.44765432098765434, 0.0, 0.0, 0.17123456790123456, 0.0, 0.18024691358024691, 0.17629629629629628], [0.6237037037037036, 0.6214814814814815, 0.4411111111111111, 0.6195061728395062, 0.44925925925925925, 0.4434567901234568, 0.0, 0.17382716049382715, 0.1725925925925926, 0.1717283950617284], [0.6324691358024691, 0.44814814814814813, 0.43296296296296294, 0.4534567901234568, 0.17987654320987653], [0.4345679012345679]]
    #arr = [[[True, False, False, False, False], [False, True, False, False, False], [False, False, True, False, False], [False, False, False, True, False], [False, False, False, False, True]], [[True, True, False, False, False], [True, False, True, False, False], [True, False, False, True, False], [True, False, False, False, True], [False, True, True, False, False], [False, True, False, True, False], [False, True, False, False, True], [False, False, True, True, False], [False, False, True, False, True], [False, False, False, True, True]], [[True, True, True, False, False], [True, True, False, True, False], [True, True, False, False, True], [True, False, True, True, False], [True, False, True, False, True], [True, False, False, True, True], [False, True, True, True, False], [False, True, True, False, True], [False, True, False, True, True], [False, False, True, True, True]], [[True, True, True, True, False], [True, True, True, False, True], [True, True, False, True, True], [True, False, True, True, True], [False, True, True, True, True]], [[True, True, True, True, True]]]
    """arr = [[0.5383115942028985, 0.5830797101449275, 0.5204782608695652, 0.5342681159420289, 0.2313913043478261], [0.5367463768115942, 0.5385869565217392, 0.5355579710144928, 0.38208695652173913, 0.5595507246376812, 0.5821159420289855, 0.5822391304347826, 0.5191014492753623, 0.42236231884057973, 0.534840579710145], [0.5380797101449275, 0.5371014492753623, 0.3831159420289855, 0.5384130434782609, 0.38407971014492753, 0.3839420289855072, 0.5605507246376812, 0.5585434782608696, 0.5844130434782608, 0.4196521739130435], [0.5396739130434782, 0.38504347826086954, 0.3832463768115942, 0.38328260869565217, 0.5578623188405797], [0.38370289855072465]]
    arr= [[[True, False, False, False, False], [False, True, False, False, False], [False, False, True, False, False], [False, False, False, True, False], [False, False, False, False, True]], [[True, True, False, False, False], [True, False, True, False, False], [True, False, False, True, False], [True, False, False, False, True], [False, True, True, False, False], [False, True, False, True, False], [False, True, False, False, True], [False, False, True, True, False], [False, False, True, False, True], [False, False, False, True, True]], [[True, True, True, False, False], [True, True, False, True, False], [True, True, False, False, True], [True, False, True, True, False], [True, False, True, False, True], [True, False, False, True, True], [False, True, True, True, False], [False, True, True, False, True], [False, True, False, True, True], [False, False, True, True, True]], [[True, True, True, True, False], [True, True, True, False, True], [True, True, False, True, True], [True, False, True, True, True], [False, True, True, True, True]], [[True, True, True, True, True]]]

    for x in range(len(arr)):
        for y in range(len(arr[x])):
            arr[x][y] = str(arr[x][y])

    for x in range(len(arr)):
        for y in range(len(arr[x])):
            #arr[x][y] = arr[x][y].replace('.', ',')
            print(arr[x][y])"""
    if evalBest is False:
        if dataSetDone:
            e.makeDatasetFromText(emotions)
        else:
            e.makeDatasetFromSound(emotions, noiseRange[0], voiceRange[0], chunkRange[0])

    #else:
    #evalBestVoiceRange(iterations, emotions, methods, noiseRange[0], voiceRange, chunkRange[0])
        # e valBestNoiseRange(iterations, emotions, methods,noiseRange, voiceRange[0], chunkRange[0])
    #evalBestMethod(iterations, emotions, methods)
        # evalBestChunkSize(iterations, emotions, methods, noiseRange[0], voiceRange[0], chunkRange)
    #evalBestMethod(iterations, emotions, methods)
    x,y = GetMeanAccuracy(iterations, emotions, methods)
    print("Accuracy: " + str(accuracy_score(x,y)))
    #evalBestEmotion(iterations, emotions, methods)
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
