import serial
import matplotlib.pyplot as plt
import numpy as np
from drawnow import *


class SensorDataManager:
    dataArray = []
    sampleSize = 0
    counter = 0

    def __init__(self, PORT, BAUDRATE, samplesize):
        print(f"Connecting to port: {PORT}...")
        self.arduino = serial.Serial(f'{PORT}', BAUDRATE)
        print(f"Connected to Arduino port:{PORT}")
        self.sampleSize = samplesize

    def plotData(self):
        plt.ion()
        plt.ylim(0, 100)
        plt.grid(True)
        plt.plot(self.dataArray, '-', label="Decibel (dB)")


    def updatePlot(self):
        while self.arduino.inWaiting() == 0:
            pass
        print("Collecting sensor data...")
        while self.counter <= self.sampleSize:
            data = self.arduino.readline()
            decode = data.decode("utf-8")
            decodedData = int(decode)
            self.dataArray.append(decodedData)
            drawnow(self.plotData)
            self.counter += 1
        print("Done collecting")
        sensorCollector.writeToCSV("test")

    def writeToCSV(self, filename):
        fileName = f"{filename}.csv"
        file = open(fileName, "w")
        print(f"Opened file: {fileName} for writing")
        print("Writing...")
        file.write(str(self.dataArray))
        print("Writing stopped")
        file.close()
        print(f"{self.sampleSize} samples saved to file: {fileName}")


if __name__ == "__main__":
    sensorCollector = SensorDataManager("COM3", 115200, 50)
    sensorCollector.plotData()
    sensorCollector.updatePlot()

