import numpy as np
import random


class PitchVariability:
    tarr = ['N', 1, 'N', 2]
    noiseValue = 0

    def getTrimmedArray(self, data):
        noisyArray = np.asarray(data)
        nparray = np.array([i for i in noisyArray if i != self.noiseValue])
        return nparray

    def isRobotTalking(self, data):
        array = np.asarray(data)
        for i in range(len(array) - 1):
            if array[i] == 'NOISE':
                continue
            else:
                print(array[i])

    def calculateVariance(self, data):
        array = np.asarray(data)
        for i in range(len(array) - 1):
            if array[i] < array[i + 1]:
                print(array[i])
                print(array[+1])
                print(True)
            else:
                print(array[i])
                print(array[+1])
                print(False)


# [random.randint(100, 300) for i in range(100)]

if __name__ == '__main__':
    pv = PitchVariability()
    # pv.calculateVariance(pv.getTrimmedArray(['NOISE', 'NOISE', 'NOISE', 'NOISE', 'NOISE', 'NOISE', 'NOISE', 101,
    # 102, 103, 104, 105, 106, 102, 104, 104, 'NOISE', 'NOISE', 'NOISE', 'NOISE']))
    print(pv.isRobotTalking(
        ['NOISE', 'NOISE', 'NOISE', 'NOISE', 'NOISE', 'NOISE', 'NOISE',
         'NOISE', 'NOISE', 'NOISE', 'NOISE']))
