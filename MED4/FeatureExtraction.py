import pyaudio
import numpy as np
import sys
import audioop
import math
import os
import wave
import serial
from scipy.io import wavfile
from FeatureSpace import FeatureSpace


class FeatureExtraction:

    def __init__(self):
        np.set_printoptions(threshold=sys.maxsize)

        self.CHUNK = 1024*2  # number of data points to read at a time
        self.RATE = 44100  # time resolution of the recording device (Hz)
        self.TARGET = 2100  # show only this one frequency

        self.samplingFreq = self.RATE
        self.minDigPitch = 50 * 2 * np.pi / self.samplingFreq  # radians/sample
        # print("minPitch", self.minDigPitch)
        self.maxDigPitch = 1000 * 2 * np.pi / self.samplingFreq  # radians/sample
        # print("maxPitch", self.maxDigPitch)

        self.fe = FeatureSpace()

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
        return estDigPitch, periodGrid, normAutoCorr

    def get_features_live(self):
        #ser = serial.Serial("COM3", 9600) comment in when arduino is in use
        voiceCounter = 0
        pitchArr = np.array([])
        decibelArr = np.array([])
        noiseCounter = 0
        dBArr = np.array([])
        p = pyaudio.PyAudio()  # start the PyAudio class
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True,
                        frames_per_buffer=self.CHUNK)  # uses default input device

        while True:
            data = np.frombuffer(stream.read(self.CHUNK), dtype=np.int16)
            decibel, pi = self.get_features_from_segment(data)
            
            dBArr = np.append(dBArr, decibel)
            with open("fakintextfile.txt", "w") as file:
                file.write(str(dBArr))

            if decibel > 59:
                #print('The estimated pitch is {0:.2f} Hz.'.format(pi))
                voiceCounter += 1
                pitchArr = np.append(pitchArr, pi)
                decibelArr = np.append(decibelArr, decibel)
                noiseCounter = 0
            else:
                noiseCounter += 1
                if noiseCounter > 3:
                    #print("ikke lyd")
                    if voiceCounter > 3:

                        pitch, dB, pitchVar, dBVar = self.get_features_from_arrays(pitchArr, decibelArr)
                        emotion = self.fe.checkEmotion([pitch,pitchVar,dBVar,dB])
                        print("Feature values (P, dB, PV, dBV):", pitch, dB, pitchVar, dBVar)


                        #ser.write(f"{emotion}".encode()) comment in when arduino is in use
                        #print("Value returned:", ser.read().decode()) comment in when arduino is in use

                    pitchArr = np.array([])
                    decibelArr = np.array([])
                    voiceCounter = 0
                    noiseCounter = 0

    def get_features_from_arrays(self, arrP, arrSL):
        pitch = np.mean(arrP)
        dB = np.mean(arrSL)
        pitchVar = np.std(arrP)
        dBVar = np.std(arrSL)
        return pitch, dB, pitchVar, dBVar

    def get_features_from_segment(self, data):
        rms = audioop.rms(data, 2)  # Root Mean Square to get volume
        decibel = 20 * math.log10(rms)

        guitarSignal = data / 2 ** 11  # normalise
        estDigPitch, periodGrid, normAutoCorr = self.combFilterPitchEstimation(guitarSignal, self.minDigPitch,
                                                                               self.maxDigPitch)
        pitch = (estDigPitch * self.samplingFreq / (2 * np.pi))
        return decibel, pitch

    def get_features_from_clip(self, soundDirectory, fileName):
        print("Starting getting features from " + fileName)
        pitch, dB, pitchVar, dBVar = [],[],[],[]

        voiceCounter = 0
        pitchArr = np.array([])
        decibelArr = np.array([])
        noiseCounter = 0

        soundFilesSpec = os.path.join(soundDirectory, fileName)
        samplerate, data = wavfile.read(soundFilesSpec)

        newArr = []
        chunk = self.CHUNK

        for x in range(int(len(data) / chunk)):
            if ((x + 1) * chunk + chunk) < len(data):
                newArr.append(data[x * chunk:(x + 1) * chunk - 1])
            else:
                newArr.append(data[x * chunk:len(data) - 1])

        #print(len(newArr))

        # create a numpy array holding a single read of audio data
        for i in range(len(newArr)):  # to it a few times just to see
            decibel, p = self.get_features_from_segment(newArr[i])

            if 59 < decibel:
                voiceCounter += 1
                pitchArr = np.append(pitchArr, p)
                decibelArr = np.append(decibelArr, decibel)
                noiseCounter = 0
            else:
                noiseCounter += 1
                if noiseCounter > 3:
                    if voiceCounter > 3:
                        p, s, pVar, sVar = self.get_features_from_arrays(pitchArr, decibelArr)

                        pitch.append(p)
                        dB.append(s)
                        pitchVar.append(pVar)
                        dBVar.append(sVar)

                    pitchArr = np.array([])
                    decibelArr = np.array([])
                    voiceCounter = 0
                    noiseCounter = 0
        return [pitch, dB, pitchVar, dBVar]

f = FeatureExtraction()
f.get_features_live()
