import numpy as np
from scipy.io import wavfile

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
    print("Total amount of samples:", len(data))
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


def mostPowerfulFrequency1(data1, samplerate1):
    data, samplerate = data1, samplerate1
    length = len(data) / samplerate
    time = np.linspace(0., length,
                       len(data))
    f = data
    dt = time[4] - time[3]
    n = len(time)
    fhat = np.fft.fft(f, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n) * np.arange(n))
    L = np.arange(1, np.floor(n / 2),
                  dtype="int")
    indices = PSD > max(PSD) * 0.3
    print("fff", indices)
    PSDclean = PSD * indices
    print('len', len(PSDclean))
    for i in range(len(freq[L])):
        if PSDclean[i] == max(PSDclean[L]):
            print("--------------------------------------------------")
            print("Most powerful frequency in the power spectrum:", freq[i])
            pwrFreq = freq[i]
            return pwrFreq


def mostPowerfulFrequency(data1, samplerate1):
    data, samplerate = data1, samplerate1
    length = len(data) / samplerate
    time = np.linspace(0., length,
                       len(data))
    f = data
    dt = time[4] - time[3]
    n = len(time)
    fhat = np.fft.fft(f, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n) * np.arange(n))
    L = np.arange(1, np.floor(n / 2),
                  dtype="int")
    indices = PSD > max(PSD) * 0.6
    print(indices == True)
    PSDclean = PSD * indices  # Zero out all others
    print('maxv', max(PSDclean[:1000]))
    print('len', len(PSDclean)/3)

    maxV = max(PSDclean[:1000])
    index = np.where(PSDclean == maxV)
    pwrFreq = min(freq[index])

    print("pwrFreq: ", pwrFreq)
    return pwrFreq


if __name__ == "__main__":
    samplerate, data = wavfile.read("Sound_Files/Emotions/Angry/Angry_1.wav")
    print(np.shape(data))

    datacut = []

    for i in range(int(len(data))):
        datacut.append(data[i][0])
    print(len(datacut))

    mostPowerfulFrequency(datacut, samplerate)

