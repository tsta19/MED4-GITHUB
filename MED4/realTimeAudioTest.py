import pyaudio
import numpy as np

np.set_printoptions(suppress=True)  # don't use scientific notation

CHUNK = 4096  # number of data points to read at a time
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


# create a numpy array holding a single read of audio data
for i in range(500):  # to it a few times just to see
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

    guitarSignal = data / 2 ** 11  # normalise
    estDigPitch, periodGrid, normAutoCorr = combFilterPitchEstimation(guitarSignal, minDigPitch, maxDigPitch)
    if 55 < (estDigPitch * samplingFreq / (2 * np.pi)) < 175:
        print('The estimated pitch is {0:.2f} Hz.'.format(estDigPitch * samplingFreq / (2 * np.pi)))

# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()

"""import sounddevice as sd
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
