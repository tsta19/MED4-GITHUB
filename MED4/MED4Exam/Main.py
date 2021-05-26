# import
from MED4Exam.evaluation import Evaluation
from MED4Exam.FeatureExtraction import FeatureExtraction
from MED4Exam.FeatureSpace import FeatureSpace
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
    methods = [True, True, True, True, True]
    methodNames = ["Pitch", "Sound Level", "Pitch Variance", "Sound Level Variance", "Power Frequency"]
    emotions = ["Happy", "Sad", "Angry", "Fear"]

    eval = False
    dataSetDone = True
    evalBest = 0

    noiseRange = [3, 6]
    voiceRange = [1, 6]
    chunkRange = [1024, 13]
    iterations = 100

    fe = FeatureExtraction()
    e = Evaluation(fe)

    if dataSetDone and evalBest == 0:
        e.makeDatasetFromText(emotions)
    elif evalBest == 0:
        e.makeDatasetFromSound(emotions, noiseRange[0], voiceRange[0], chunkRange[0])

    if eval:
        if evalBest == 1:
            evalBestChunkSize(iterations, emotions, methods, noiseRange[0], voiceRange[0], chunkRange)
        elif evalBest == 2:
            evalBestNoiseRange(iterations, emotions, methods, noiseRange, voiceRange[0], chunkRange[0])
        elif evalBest == 3:
            evalBestVoiceRange(iterations, emotions, methods, noiseRange[0], voiceRange, chunkRange[0])
        elif evalBest == 4:
            evalBestEmotion(iterations, emotions, methods)
        elif evalBest == 5:
            evalBestMethod(iterations, emotions, methods)

        x,y = GetMeanAccuracy(iterations, emotions, methods)
        print("Accuracy: " + str(accuracy_score(x,y)))
    else:
        fe.get_features_live(emotions, methods, noiseRange[0], voiceRange[0], chunkRange[0])
