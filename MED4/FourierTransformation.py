import contextlib
import math
import os
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa as lbs
from scipy.io.wavfile import write

# np.set_printoptions(threshold=sys.maxsize)

wav_fname = "testest.wav"
data, samplerate = lbs.load(wav_fname)

print(f'Samplerate = {samplerate}')

length = data.shape[0] / samplerate
print(f"length = {length}s")

time = np.linspace(0., length, data.shape[0])  # list the size of samplesize with 1 sample-time length per iteration
f = data  # Signal
dt = time[4] - time[3]  ##Iteration length variable
n = len(time)  ##Amount of samples
fhat = np.fft.fft(f, n)  ### Fourier transformed signal
PSD = fhat * np.conj(fhat) / n  ## Computing power spectrum of the signal
freq = (1 / (dt * n) * np.arange(n))  ## Making freqeuncies for x-axis
L = np.arange(1, np.floor(n / 2), dtype="int")  ## Only plot the first half of freqs, this seperates the second half
indices = PSD > max(PSD)*0.7  # Find all freqs with large power
for i in range(len(indices)):
    if indices[i] == True:
        print(indices[i])
        print(freq[i])
print("denne her lægnde", len(freq[L]))
PSDclean = PSD * indices  # Zero out all others
fhat = indices * fhat  # Zero out small Fourier coeffs. in Y
ffilt = np.fft.ifft(fhat)
absFfilt = ffilt.real

def getFreqDistribution(data):
    freq01 = []
    freq12 = []
    freq23 = []
    freq34 = []
    freq45 = []
    freq56 = []
    freq67 = []
    freq78 = []
    freq89 = []
    freq91 = []
    freq1011 = []
    freq1112 = []

    for j in range(len(data)):
        if 1 < data[j] * 10000 < 100:
            freq01.append(data[j])

        if 100 < data[j] * 10000 < 200:
            freq12.append(data[j])

        if 200 < data[j] * 10000 < 300:
            freq23.append(data[j])

        if 300 < data[j] * 10000 < 400:
            freq34.append(data[j])

        if 400 < data[j] * 10000 < 500:
            freq45.append(data[j])

        if 500 < data[j] * 10000 < 600:
            freq56.append(data[j])

        if 600 < data[j] * 10000 < 700:
            freq67.append(data[j])

        if 700 < data[j] * 10000 < 800:
            freq78.append(data[j])

        if 800 < data[j] * 10000 < 900:
            freq89.append(data[j])

        if 900 < data[j] * 10000 < 1000:
            freq91.append(data[j])

        if 1000 < data[j] * 10000 < 1100:
            freq1011.append(data[j])

        if 1100 < data[j] * 10000 < 1200:
            freq1112.append(data[j])
    print("--------------------------------------------------")
    print("Samples in frequency range 0-100:", len(freq01))
    print("Samples in frequency range 100-200:", len(freq12))
    print("Samples in frequency range 200-300:", len(freq23))
    print("Samples in frequency range 300-400:", len(freq34))
    print("Samples in frequency range 400-500:", len(freq45))
    print("Samples in frequency range 500-600:", len(freq56))
    print("Samples in frequency range 600-700:", len(freq67))
    print("Samples in frequency range 700-800:", len(freq78))
    print("Samples in frequency range 800-900:", len(freq89))
    print("Samples in frequency range 900-1000:", len(freq91))
    print("Samples in frequency range 1000-1100:", len(freq1011))
    print("Samples in frequency range 1100-1200:", len(freq1112))
    print("--------------------------------------------------")


if __name__ == "__main__":
    fig, axis = plt.subplots(3, 1)
    #
    plt.sca(axis[0])
    plt.plot(time, f, color='c', LineWidth=1.5, label="Sample")
    plt.xlim(time[0], time[-1])
    plt.ylabel("Amplitude [?]")
    plt.xlabel("Seconds [s]")
    plt.legend()

    plt.sca(axis[1])
    plt.plot(time, absFfilt, color="royalblue", label="Filtered")
    plt.xlim(time[0], time[-1])
    plt.ylabel("Amplitude [?]")
    plt.xlabel("Seconds [s]")
    plt.legend()

    plt.sca(axis[2])
    plt.plot(time, f - absFfilt, color="mediumslateblue", label="Sample - Filtered")
    plt.xlim(time[0], time[-1])
    plt.ylabel("Amplitude [?]")
    plt.xlabel("Seconds [s]")
    plt.legend()
    plt.show()

    # plt.sca(axis[0])
    plt.plot(freq[L], PSD[L], color="r", LineWidth=2, label="Noisy")
    plt.plot(freq[L], PSDclean[L], color="c", LineWidth=2, label="Filtered")
    plt.xlim(freq[L[0]], 1000)
    plt.ylabel("Power")
    plt.xlabel("Frequency [Hz]")
    plt.legend()
    plt.show()

    write("yyyyxu.wav", 22050, absFfilt)
    print("--------------------------------------------------")
    print("Mean frequency:", np.mean(abs(absFfilt)) * 10000)
    print("Median frequency:", np.median(abs(absFfilt))*10000)
    print("Maximum frequency:", np.max(abs(absFfilt)) * 10000)
    print("Minimum frequency:", np.min(abs(absFfilt)) * 10000)


    for i in range(len(freq[L])):
        if PSDclean[i] == max(PSDclean[L]):
            print("--------------------------------------------------")
            print("Most powerful frequency in the power spectrum:", freq[i])
            break
