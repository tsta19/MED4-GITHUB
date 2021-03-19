import numpy as np
import time


class QuickSort:

    def quickSort(self, sequence):
        length = len(sequence)
        greaterNumbers = []
        lowerNumbers = []

        if length <= 1:
            return sequence
        else:
            pivotPoint = sequence.pop()

        for number in sequence:
            if number > pivotPoint:
                greaterNumbers.append(number)
            else:
                lowerNumbers.append(number)

        return self.quickSort(lowerNumbers) + [pivotPoint] + self.quickSort(greaterNumbers)
