from typing import Optional

import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from librosa.display import specshow
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile
from sympy.stats.drv_types import scipy


class SpectraAnalysis:
    iterator = 1
    debug = True

    def spectra(self, name_of_audio_file):
        samples, sampleRate = librosa.load(name_of_audio_file, sr=None, mono=True, offset=0.0,
                                           duration=None)
        if self.debug:
            plt.figure()
            librosa.display.waveplot(y=samples, sr=sampleRate)
            plt.xlabel("Time (Seconds)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.show()

    def fastFourierTransform(self, data, sampleRate):
        sampleLength = len(data)
        time = 1 / sampleRate
        yAxis = scipy.fft(data)
        xAxis = np.linspace(0.0, 1.0 / (2.0 * time), sampleLength // 2)
        if self.debug:
            figure, graph = plt.subplots()
            graph.plot(xAxis, 2.0 / sampleLength * np.abs(yAxis[:sampleLength // 2]))
            plt.grid()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.show()

    def spectrogram(self, samples, sample_rate, stride_ms=10.0,
                    window_ms=20.0, max_freq=22050, eps=1e-14):
        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)

        # Extract strided windows
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[:len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(samples,
                                                  shape=nshape, strides=nstrides)

        assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

        # Window weighting, squared Fast Fourier Transform (fft), scaling
        weighting = np.hanning(window_size)[:, None]

        fft = np.fft.rfft(windows * weighting, axis=0)
        fft = np.absolute(fft)
        fft = fft ** 2

        scale = np.sum(weighting ** 2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale

        # Prepare fft frequency list
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

        # Compute spectrogram feature
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        spectrogram = np.log(fft[:ind, :] + eps)

        if self.debug:
            specshow(spectrogram, x_axis="time", y_axis="hz")

    def meanFrequency(self, data: np.ndarray, samplingFrequency: int) -> float:
        spec = np.abs(np.fft.rfft(data))
        frequency = np.fft.rfftfreq(len(data), d=1 / samplingFrequency)
        amplitude = spec / spec.sum()
        meanFrequency = (frequency * amplitude).sum()
        return meanFrequency
