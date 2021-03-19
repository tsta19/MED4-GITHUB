import sys
import numpy as np
from QuickSort import *

class FileSorter():
    np.set_printoptions(threshold=sys.maxsize)
    fileName = "pitch_values.txt"
    newFileName = "sorted_pitch_values.txt"
    read = "r"
    write = "w"
    pitchValuesArray = []
    pitchArray = []


    def removeNormalZeroes(self):
        with open(self.fileName, self.read) as file:
            lines = file.readlines()

        with open(self.fileName, self.write) as file:
            for line in lines:
                if line.strip() != "0;":
                    file.write(line)
        file.close()

    def removeSpacedZeroes(self):
        with open(self.fileName, self.read) as file:
            lines = file.readlines()

        with open(self.fileName, self.write) as file:
            for line in lines:
                if line.strip() != "0 ;":
                    file.write(line)
        file.close()

    def convertToArray(self):
        with open(self.fileName, self.read) as file:
            lines = file.readlines()
            for line in lines:
                self.pitchValuesArray.append(str(line))
                # Remove the last two characters from each element in the array
                trimmedPitchArray = np.array([x[:-2] for x in self.pitchValuesArray])
                numericPitchArray = trimmedPitchArray.astype(np.float)

        return numericPitchArray

    def writeTextFile(self, value):
        file = open(self.newFileName, "w+")
        file.write(str(value))


if __name__ == '__main__':
    qs = QuickSort()
    fs = FileSorter()
    fs.removeNormalZeroes()
    fs.removeSpacedZeroes()
    sortedList = qs.quickSort(fs.convertToArray().tolist())
    fs.writeTextFile(sortedList)
