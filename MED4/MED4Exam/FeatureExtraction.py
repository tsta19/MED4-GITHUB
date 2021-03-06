import pyaudio
import numpy as np
import sys
import audioop
import math
import os
from scipy.io import wavfile
from MED4Exam.FeatureSpace import FeatureSpace

class FeatureExtraction:

    def __init__(self):
        np.set_printoptions(threshold=sys.maxsize)

        self.CHUNK = 1500  # number of data points to read at a time
        self.RATE = 44100  # time resolution of the recording device (Hz)
        self.TARGET = 2100  # show only this one frequency

        self.samplingFreq = self.RATE
        self.minDigPitch = 50 * 2 * np.pi / self.samplingFreq  # radians/sample
        self.maxDigPitch = 1000 * 2 * np.pi / self.samplingFreq  # radians/sample

        self.fs = FeatureSpace()

    def getFeatureSpace(self):
        return self.fs

    def combFilterPitchEstimation(self, inputSignal, minDigPitch, maxDigPitch):
        minPeriod = np.int(np.ceil(2 * np.pi / maxDigPitch))
        maxPeriod = np.int(np.floor(2 * np.pi / minDigPitch))
        periodGrid = np.arange(minPeriod, maxPeriod + 1)
        nPitches = np.size(periodGrid)
        normAutoCorr = np.zeros(nPitches)
        signal = inputSignal[maxPeriod:]
        signalPower = np.sum(signal ** 2)

        for ii in np.arange(nPitches):
            iiPeriod = periodGrid[ii]
            shiftedSignal = inputSignal[maxPeriod - iiPeriod:-iiPeriod]
            shiftedSignalPower = np.sum(shiftedSignal ** 2)
            normAutoCorr[ii] = np.max((np.sum(signal * shiftedSignal) / np.sqrt(signalPower * shiftedSignalPower), 0))

        estPeriodIdx = np.argmax(normAutoCorr)
        estDigPitch = 2 * np.pi / periodGrid[estPeriodIdx]
        return estDigPitch

    def get_features_from_arrays(self, arrP, arrSL, rawData):
        pitch = np.mean(arrP)
        dB = np.mean(arrSL)
        pitchVar = np.std(arrP)
        dBVar = np.std(arrSL)

        powerFreq = self.mostPowerfulFrequency(rawData, self.RATE)

        return pitch, dB, pitchVar, dBVar, powerFreq

    def mostPowerfulFrequency(self, data1, samplerate1):
        data, samplerate = data1, samplerate1
        length = len(data) / samplerate
        time = np.linspace(0., length, len(data))
        f = data
        dt = time[4] - time[3]
        n = len(time)
        fhat = np.fft.fft(f, n)
        PSD = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)

        indices = PSD > max(PSD) * .8
        PSDclean = PSD * indices

        maxV = max(PSDclean[:300])
        index = np.where(PSDclean[:300] == maxV)
        pwrFreq = freq[index]
        pwrFreq = pwrFreq[0]

        return pwrFreq

    def get_features_from_segment(self, data):
        rms = audioop.rms(data, 2)  # Root Mean Square to get volume
        decibel = 20 * math.log10(rms)

        signal = data / 2 ** 11  # normalise
        estDigPitch = self.combFilterPitchEstimation(signal, self.minDigPitch,
                                                                               self.maxDigPitch)
        pitch = (estDigPitch * self.samplingFreq / (2 * np.pi))
        return decibel, pitch

    def get_features_from_clip(self, soundDirectory, fileName, noiseRange, voiceRange, chunkRange):
        print("Starting getting features from " + fileName)

        voiceCounter = 0
        pitchArr = np.array([])
        decibelArr = np.array([])
        dataArr = np.array([])
        noiseCounter = 0

        soundFilesSpec = os.path.join(soundDirectory, fileName)
        samplerate, data = wavfile.read(soundFilesSpec)

        newArr = []
        chunk = chunkRange

        for x in range(int(len(data) / chunk)):
            if ((x + 1) * chunk + chunk) < len(data):
                newArr.append(data[x * chunk:(x + 1) * chunk ])

        features = np.array([[]])

        for i in range(len(newArr)):
            decibel, p = self.get_features_from_segment(newArr[i])

            if 59 < decibel and p < 300:
                voiceCounter += 1
                pitchArr = np.append(pitchArr, p)
                decibelArr = np.append(decibelArr, decibel)
                dataArr = np.append(dataArr, newArr[i])
                noiseCounter = 0
            else:
                noiseCounter += 1
                if noiseCounter > noiseRange:
                    if voiceCounter > voiceRange:
                        p, s, pVar, sVar, pFreq = self.get_features_from_arrays(pitchArr, decibelArr, dataArr)

                        print(str(round((((i)*chunk)/ self.RATE),2)) + "-" + str(round((((i+1)*chunk)/ self.RATE),2)) + " s: " + str([p, s, pVar, sVar, pFreq]))
                        if len(features) <= 1:
                            features = [p, s, pVar, sVar,pFreq]
                        else:
                            features = np.vstack((features, [p, s, pVar, sVar,pFreq]))

                    pitchArr = np.array([])
                    decibelArr = np.array([])
                    dataArr = np.array([])

                    voiceCounter = 0
                    noiseCounter = 0

        return features

    def get_features_live(self, emotions, methods, noiseRange, voiceRange, chunkRange):
        # ser = serial.Serial("COM3", 9600) comment in when arduino is in use
        voiceCounter = 0
        pitchArr = np.array([])
        decibelArr = np.array([])
        dataArr = np.array([])
        noiseCounter = 0

        p = pyaudio.PyAudio()  # start the PyAudio class
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True,
                        frames_per_buffer=self.CHUNK)  # uses default input device

        self.fs.setMethods(methods)
        self.fs.setEmotions(emotions)
        self.fs.setFeatureSpaces()

        print("Started listening...")

        while True:
            data = np.frombuffer(stream.read(self.CHUNK), dtype=np.int16)
            decibel, pi = self.get_features_from_segment(data)

            if 59 < decibel and pi < 300:
                voiceCounter += 1
                pitchArr = np.append(pitchArr, pi)
                decibelArr = np.append(decibelArr, decibel)
                dataArr = np.append(dataArr, data)
                noiseCounter = 0
            else:
                noiseCounter += 1
                if noiseCounter > noiseRange:
                    if voiceCounter > voiceRange:
                        pitch, dB, pitchVar, dBVar, pFreq = self.get_features_from_arrays(pitchArr, decibelArr, dataArr)

                        emotion = self.fs.checkEmotion([pitch, pitchVar, dBVar, dB, pFreq])

                        # ser.write(f"{emotion}".encode()) comment in when arduino is in use
                        # print("Value returned:", ser.read().decode()) comment in when arduino is in use

                    pitchArr = np.array([])
                    decibelArr = np.array([])
                    voiceCounter = 0
                    noiseCounter = 0

