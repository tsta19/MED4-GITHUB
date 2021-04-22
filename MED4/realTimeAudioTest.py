import pyaudio
import numpy as np
import sys
import audioop
import math
from scipy.io import  wavfile

np.set_printoptions(threshold=sys.maxsize)

CHUNK = 1024*2  # number of data points to read at a time
RATE = 44100  # time resolution of the recording device (Hz)
TARGET = 2100  # show only this one frequency

p = pyaudio.PyAudio()  # start the PyAudio class
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                frames_per_buffer=CHUNK)  # uses default input device

samplingFreq = RATE
minDigPitch = 50 * 2 * np.pi / samplingFreq  # radians/sample
print("minPitch", minDigPitch)
maxDigPitch = 1000 * 2 * np.pi / samplingFreq  # radians/sample
print("maxPitch", maxDigPitch)


def combFilterPitchEstimation(inputSignal, minDigPitch, maxDigPitch):
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


# 45.575087508483136 - 48.21612926584542
# 49.87701409139849 - 52.8979408854918

voiceCounter = 0
pitchArr = np.array([])
decibelArr = np.array([])
noiseCounter = 0

samplerate, data = wavfile.read('sur1.wav')
newArr = []
chunk = 1024

print(len(data))
print(int(len(data)/chunk))

for x in range(int(len(data)/chunk)):
    if ((x+1)*chunk + chunk) < len(data):
        #(((x+1)*chunk + chunk))
        newArr.append(data[x*chunk:(x+1)*chunk-1])
    else:
        newArr.append(data[x*chunk:len(data)-1])

print(len(newArr))

# create a numpy array holding a single read of audio data
for i in range(len(newArr)):  # to it a few times just to see
    #data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    data = newArr[i]

    rms = audioop.rms(data, 2)  # Root Mean Square to get volume
    decibel = 20 * math.log10(rms)

    """if 45 > decibel >= 40:
        print("Low Sound")
    elif 49 > decibel >= 45:
        print("MediumHigh Sound")
    elif decibel >= 49:
        print("High Sound")"""

    #SoundLevel = np.mean(data)
    #print('Sound Level: ' + str(SoundLevel) + " dB")

    guitarSignal = data / 2 ** 11  # normalise
    estDigPitch, periodGrid, normAutoCorr = combFilterPitchEstimation(guitarSignal, minDigPitch, maxDigPitch)
    if 55 < (estDigPitch * samplingFreq / (2 * np.pi)) < 175:
        #print('The estimated pitch is {0:.2f} Hz.'.format(estDigPitch * samplingFreq / (2 * np.pi)))
        voiceCounter += 1
        pitchArr = np.append(pitchArr, (estDigPitch * samplingFreq / (2 * np.pi)))
        decibelArr = np.append(decibelArr, decibel)
        noiseCounter = 0
    else:
        noiseCounter += 1
        if noiseCounter > 3:
        #print("ikke lyd")
            if voiceCounter > 3:
                print()
                print("Pitch: " + str(np.mean(pitchArr)))
                print("Decibel: " + str(np.mean(decibelArr)))
                print("Pitch variance: " + str(np.amax(pitchArr) - np.amin(pitchArr)))
                print("Decibel variance: " + str(np.amax(decibelArr) - np.amin(decibelArr)))

            pitchArr = np.array([])
            decibelArr = np.array([])
            voiceCounter = 0
            noiseCounter = 0


# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()


"""print(resultArr)
high = 100
low = 0
for x in range(len(resultArr)):
    if resultArr[x] > low:
        low = resultArr[x]
    if resultArr[x] < high:
        high = resultArr[x]
print("High: " + str(high))
print("Low: " + str(low))

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
from librosa import display
import librosa

fs = 44100  # Sample rate
clipMs = 500
overlapMs = 200
seconds = clipMs/1000  # Duration of recording
overlapSeconds = overlapMs/1000

arr = [1,2,3,4,5,6,7,8,9,10,11]

arr = np.delete(arr, [0,1,2,3,4,5,6,7,8,9],None)
print(arr)

print("start recording...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

sd.wait()  # Wait until recording is finished

np.set_printoptions(threshold=np.inf)
print(myrecording)
print(len(myrecording))
write('output.wav', fs, myrecording)  # Save as WAV file

time = np.linspace(0., len(myrecording)/fs, len(myrecording))

plt.grid(True)
plt.plot(time, myrecording, '-', label="Decibel (dB)")
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")

plt.show()

times = 1000
counter = 0

rangeArr = []
for x in range(4410*2):
    rangeArr.append(x)

while len(myrecording)/fs*1000 >= clipMs:

    average = 0.
    for x in range(len(myrecording)):
        average += myrecording[x]

    #print(myrecording)
    print(average)

    average = average / len(myrecording)

    print(average)

    #print(len(myrecording))
    if average > 0.05:
        print("SOUND AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    newArr = np.delete(myrecording, rangeArr, None)

    #print(newArr)
    #print(len(newArr))

    if counter < times:
        newRecording = sd.rec(int(overlapSeconds*fs), samplerate=fs, channels=1)
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        myrecording = np.append(newArr, newRecording)
        counter += 1
        #print(newRecording)
        #print(myrecording)
    else:
        break

print("done")
"""
