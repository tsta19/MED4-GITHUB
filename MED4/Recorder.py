import pyaudio  # Soundcard audio I/O access library
import wave  # Python 3 module for reading / writing simple .wav files
from Explorer import *

class Recorder:
    # Setup channel info
    FORMAT = pyaudio.paInt16  # data type formate
    CHANNELS = 1  # Adjust to your number of channels
    RATE = 44100  # Sample Rate
    CHUNK = 1024  # Block Size
    RECORD_SECONDS = 3  # Record time
    WAVE_OUTPUT_FILENAME = "sound_file_1.wav"

    # Startup pyaudio instance
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Started recording for {} second(s)".format(RECORD_SECONDS))
    frames = []

    # Record for RECORD_SECONDS
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording has Finished")

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    exp = Explorer()

    # Write your new .wav file with built in Python 3 Wav module
    waveFile = wave.open(exp.getAudioFilePath() + WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
