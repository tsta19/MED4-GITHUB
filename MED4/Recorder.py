import pyaudio  # Soundcard audio I/O access library
import wave  # Python 3 module for reading / writing simple .wav files
from Explorer import *


class Recorder:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLERATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "sound_file_1.wav"

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLERATE, input=True, frames_per_buffer=CHUNK)

    print("Started recording for {} second(s)".format(RECORD_SECONDS))
    frames = []

    for i in range(0, int(SAMPLERATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording has Finished")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    exp = Explorer()

    # Write your new .wav file with built in Python 3 Wav module
    waveFile = wave.open(exp.getAudioFilePath() + WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(SAMPLERATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
