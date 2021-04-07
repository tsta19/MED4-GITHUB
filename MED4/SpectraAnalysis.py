import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile


class SpectraAnalysis:
    iterator = 1

    def spectra(self, name_of_audio_file):
        sampleRate, data = wavfile.read(name_of_audio_file)
        samples = data.shape[0]
        plt.plot(data)
        plt.title('Signal over Samples, Picture {}'.format(self.iterator))
        plt.xlabel('Samples')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()
        dataRFFT = fft(data)
        # Get the absolute value of real and complex component:
        fftABS = abs(dataRFFT)
        freqs = fftfreq(samples, 1 / sampleRate)
        plt.plot(freqs, fftABS)
        plt.title('Signal w/ Fast Fourier Transform, Picture {}'.format(self.iterator))
        plt.xlim([10, sampleRate / 2])
        plt.xscale('log')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.plot(freqs[:int(freqs.size / 2)], fftABS[:int(freqs.size / 2)])
        plt.tight_layout()
        plt.show()

    def meanFrequency(self, y: np.ndarray, fs: int) -> float:
        spec = np.abs(np.fft.rfft(y))
        frequency = np.fft.rfftfreq(len(y), d=1 / fs)
        amplitude = spec / spec.sum()
        meanFrequency = (frequency * amplitude).sum()
        return meanFrequency