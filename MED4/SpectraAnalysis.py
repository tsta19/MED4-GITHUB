import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile


class SpectraAnalysis:

    def spectra(self, name_of_audio_file):
        sampleRate, data = wavfile.read(name_of_audio_file)
        samples = data.shape[0]
        plt.plot(data)
        plt.title('Signal over Samples')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (dB)')
        plt.show()
        dataRFFT = fft(data)
        # Get the absolute value of real and complex component:
        fftABS = abs(dataRFFT)
        freqs = fftfreq(samples, 1 / sampleRate)
        plt.plot(freqs, fftABS)
        plt.title('Signal w/ Fast Fourier Transform')
        plt.xlim([10, sampleRate / 2])
        plt.xscale('log')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.plot(freqs[:int(freqs.size / 2)], fftABS[:int(freqs.size / 2)])
        plt.show()

    def spectral_statistics(self, y: np.ndarray, fs: int) -> float:
        spec = np.abs(np.fft.rfft(y))
        frequency = np.fft.rfftfreq(len(y), d=1 / fs)
        amplitude = spec / spec.sum()
        meanFrequency = (frequency * amplitude).sum()
        return meanFrequency

        # def spectral_properties(y: np.ndarray, fs: int) -> dict:
    #     spec = np.abs(np.fft.rfft(y))
    #     freq = np.fft.rfftfreq(len(y), d=1 / fs)
    #     spec = np.abs(spec)
    #     amp = spec / spec.sum()
    #     mean = (freq * amp).sum()
    #     sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    #     amp_cumsum = np.cumsum(amp)
    #     median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    #     mode = freq[amp.argmax()]
    #     Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    #     Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    #     IQR = Q75 - Q25
    #     z = amp - amp.mean()
    #     w = amp.std()
    #     skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    #     kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
    #
    #     result_d = {
    #         'mean': mean,
    #         'sd': sd,
    #         'median': median,
    #         'mode': mode,
    #         'Q25': Q25,
    #         'Q75': Q75,
    #         'IQR': IQR,
    #         'skew': skew,
    #         'kurt': kurt
    #     }
    #
    #     return result_d
