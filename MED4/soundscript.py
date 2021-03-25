from scipy.io import wavfile
import sys
import numpy
import matplotlib.pyplot as plt
import numpy as np

numpy.set_printoptions(threshold=sys.maxsize)

wav_fname = "bonk.wav"
samplerate, data = wavfile.read(wav_fname)
print(f"number of channels = {data.shape[1]}")

print(f'Samplerate = {samplerate}')

length = data.shape[0] / samplerate
print(f"length = {length}s")

time = np.linspace(0., length, data.shape[0])
plt.plot(time, data[:, 0], label="Left channel")
plt.plot(time, data[:, 1], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
