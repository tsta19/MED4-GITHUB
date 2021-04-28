import sys

import librosa as lbs
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter

np.set_printoptions(threshold=sys.maxsize)


class NoiseReductionNew:

    def bandpass(self, lowCut, highCut, fs, order=5):
        nyq = 0.5 * fs
        low = lowCut / nyq
        high = highCut / nyq
        b = butter(order, [low, high], btype='band')
        return b

    def lowpass(self, cutoffFrequency, fs, order=5):
        nyq = 0.5 * fs
        low = cutoffFrequency / nyq
        b = butter(order, low, btype='low')
        return b

    def bandpass_filter(self, data, lowCut, highCut, fs, order=5):
        b, a = self.bandpass(lowCut, highCut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def lowpass_filter(self, data, cutoffFrequency, fs, order=5):
        b, a = self.lowpass(cutoffFrequency, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def noiseRemover(self, data, cutoffValue):
        array = np.asarray(data)
        newArray = np.array([i for i in array if i > cutoffValue])
        return newArray


if __name__ == '__main__':
    nrn = NoiseReductionNew()

    sampleData, sampleRate = lbs.load('Wav/sound_clip.wav')
    noiseData, noiseRate = lbs.load('Wav/bgnosie.wav')

    filteredAudio = nrn.bandpass_filter(sampleData, 95, 1500, sampleRate, 1)
    wavfile.write('Wav/filtered_angry1.wav', 22050, filteredAudio)


    length = sampleData.shape[0] / sampleRate
    time = np.linspace(0., length, sampleData.shape[0])  # list the size of samplesize with 1 sample-time length per iteration
    dt = time[4] - time[3]  ##Iteration length variable
    n = len(time)  ##Amount of samples
    f = sampleData  # Signal
    fhat = np.fft.fft(f, n)  ### Fourier transformed signal
    PSD = fhat * np.conj(fhat) / n  ## Computing power spectrum of the signal
    L = np.arange(1, np.floor(n / 2), dtype="int")  ## Only plot the first half of freqs, this seperates the second half
    indices = PSD > 2  # Find all freqs with large power
    PSDclean = PSD * indices  # Zero out all others
    ffilt = np.fft.ifft(fhat)
    absFfilt = ffilt.real

    freq = (1 / (dt * n) * np.arange(n))  ## Making freqeuncies for x-axis
    fig, axis = plt.subplots(3, 1)
    plt.sca(axis[0])
    plt.plot(time, f, color='c', LineWidth=1.5, label="Sample")
    plt.xlim(time[0], time[-1])
    plt.ylabel("Amplitude [?]")
    plt.xlabel("Seconds [s]")
    plt.legend()

    plt.sca(axis[1])
    plt.plot(time, ffilt, color="royalblue", label="Filtered")
    plt.xlim(time[0], time[-1])
    plt.ylabel("Amplitude [?]")
    plt.xlabel("Seconds [s]")
    plt.legend()

    plt.sca(axis[2])
    plt.plot(time, absFfilt, color="mediumslateblue", label="Sample - Filtered")
    plt.xlim(time[0], time[-1])
    plt.ylabel("Amplitude [?]")
    plt.xlabel("Seconds [s]")
    plt.legend()
    plt.tight_layout()
    plt.show()
