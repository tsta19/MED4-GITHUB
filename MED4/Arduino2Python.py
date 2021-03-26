import serial
import matplotlib.pyplot as plt
import numpy as np
from drawnow import *


class SensorDataManager:
    dataArray = []
    sampleSize = 0
    counter = 0
    baud = 0
    port = 0
    arduino = 0

    def __init__(self, PORT, BAUDRATE, samplesize):
        self.port = PORT
        self.baud = BAUDRATE
        self.sampleSize = samplesize

    def plotData(self):
        plt.ion()
        plt.ylim(int(np.min(self.dataArray) - 5), int(np.max(self.dataArray) + 5))
        plt.grid(True)
        plt.plot(self.dataArray, '-', label="Decibel (dB)")

    def drawPlot(self):
        drawnow(self.plotData)

    def writeToCSV(self, filename, data):
        fileName = f"{filename}.csv"
        file = open(fileName, "w")
        print(f"Opened file: {fileName} for writing")
        print("Writing...")
        file.write(str(data))
        print("Writing stopped")
        file.close()
        print(f"{self.sampleSize} samples saved to file: {fileName}")

    def collectData(self):
        print(f"Connecting to port: {self.port}...")
        self.arduino = serial.Serial(f'{self.port}', self.baud)
        print(f"Connected to Arduino port:{self.port}")
        while self.arduino.inWaiting() == 0:
            pass
        print("Collecting sensor data...")
        while self.counter <= self.sampleSize:
            data = self.arduino.readline()
            decode = data.decode("utf-8")
            decodedData = int(decode)
            self.dataArray.append(decodedData)
            self.counter += 1

        return self.dataArray


if __name__ == "__main__":
    sensorCollector = SensorDataManager("COM3", 9600, 1000)
    sensorCollector.writeToCSV("test", sensorCollector.collectData())
    sensorCollector.plotData()
    sensorCollector.drawPlot()
