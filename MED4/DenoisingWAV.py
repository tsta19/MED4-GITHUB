import contextlib
import math
import os
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa as lbs

# np.set_printoptions(threshold=sys.maxsize)

wav_fname = "testest.wav"
data, samplerate = lbs.load(wav_fname)

print(f'Samplerate = {samplerate}')

length = data.shape[0] / samplerate
print(f"length = {length}s")

time = np.linspace(0., length, data.shape[0])  # list the size of samplesize with 1 sample-time length per iteration
f = data  # Signal
print("f:", f)
dt = time[4] - time[3]  ##Iteration length variable
print("dt", dt)
n = len(time)  ##Amount of samples
print("n", n)
fhat = np.fft.fft(f, n)  ### Fourier transformed signal
PSD = fhat * np.conj(fhat) / n  ## Computing power spectrum of the signal
freq = (1 / (dt * n) * np.arange(n))  ## Making freqeuncies for x-axis
print("nparange", np.arange(n))
print("matth", (1 / (dt * n)))
print("freq", freq)
L = np.arange(1, np.floor(n / 2), dtype="int")  ## Only plot the first half of freqs, this seperates the second half
print("what", L)
indices = PSD > 2  # Find all freqs with large power
PSDclean = PSD * indices  # Zero out all others
fhat = indices * fhat  # Zero out small Fourier coeffs. in Y
ffilt = np.fft.ifft(fhat)
absFfilt = ffilt.real

print("absfilt", absFfilt)
print(absFfilt.dtype)


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
    if sample_width == 1:
        dtype = np.uint8  # unsigned char
    elif sample_width == 2:
        dtype = np.int16  # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels


def noiseFiltering(soundFile, wavfile):
    with contextlib.closing(wave.open(wavfile, 'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames * nChannels)
        spf.close()
        # channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

        wav_file = wave.open("newfile.wav", "w")
        wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(soundFile.tobytes('C'))
        wav_file.close()


if __name__ == "__main__":
    pass

    noiseFiltering(ffilt, wav_fname)
    fig, axis = plt.subplots(4, 1)

    plt.sca(axis[0])
    plt.plot(time, f, color='c', LineWidth=1.5, label="Sample")
    plt.xlim(time[0], time[-1])
    plt.ylabel("Magnitude [?]")
    plt.xlabel("Seconds [s]")
    plt.legend()

    plt.sca(axis[1])
    plt.plot(freq[L], PSD[L], color="r", LineWidth=2, label="Noisy")
    plt.plot(freq[L], PSDclean[L], color="c", LineWidth=2, label="absFiltered")
    plt.xlim(freq[L[0]], 1000)
    plt.ylabel("Power")
    plt.xlabel("Frequency [Hz]")
    plt.legend()

    plt.sca(axis[2])
    plt.plot(time, absFfilt, color="c", label="Filtered")
    plt.xlim(time[0], time[-1])
    plt.legend()

    plt.sca(axis[3])
    plt.plot(time, ffilt, color="c", label="Filtered")
    plt.xlim(time[0], time[-1])
    plt.legend()
    plt.show()
