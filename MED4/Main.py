# imports
import glob
import os
from Recorder import *
from DataManager import *
from NoiseReduction import *
from Explorer import *
from SpectraAnalysis import *
from VoiceRecognizer import *

if __name__ == '__main__':
    # Modes for processing
    # Single File is a manual version where file name and outputName is specified
    # Directory is an automatic version that goes through files in a directory
    modeSingleFile = False
    modeDirectory = True

    # Iterator(s)
    iteration = 1

    # Class instantiations
    exp = Explorer()
    rec = Recorder()
    nRed = NoiseReduction()
    spectra = SpectraAnalysis()
    vr = VoiceRecognizer()

    if modeSingleFile:
        number = 1
        soundFile = "{}sound_file_{}.{}".format(exp.getAudioFilePath(), number, "wav")
        singleFileSpecification = os.path.join(str(soundFile))
        nRed.noiseFiltering(False, soundFile, singleFileSpecification)
        spectra.spectra("sound_file_1")
        sampleRate, data = wavfile.read(exp.getAudioFilePath() + "sound_file_1" + ".wav")
        vr.recognize(spectra.spectral_statistics(data, 3000))
        print("Mean Frequency:", spectra.spectral_statistics(data, 3000))

    if modeDirectory:
        soundFile = "noise"
        outputName = "filtered"
        for filename in glob.glob(os.path.join(exp.getAudioFilePath(), '*.wav')):
            wavFile = wave.open(filename, 'r')
            data = wavFile.readframes(wavFile.getnframes())
            wavFile.close()

        for audioFiles in glob.glob(os.path.join(exp.getAudioFilePath() + str(soundFile) + '*.wav')):
            print("--------------------------")
            print("For Loop iteration:", iteration)

            audioFileSpecification = os.path.join(exp.getAudioFilePath(), str(soundFile) + "_" + str(iteration) + "_" + str(outputName) + ".wav")
            print("File:", audioFileSpecification)
            nRed.noiseFiltering(True, soundFile, audioFileSpecification, audioFiles)
            spectra.spectra(audioFileSpecification)
            sampleRate, data = wavfile.read(exp.getAudioFilePath() + soundFile + "_" + str(iteration) + "_"
                                            + str(outputName) + ".wav")
            meanFrequency = spectra.spectral_statistics(data, 5000)
            vr.recognize(meanFrequency)
            print("Mean Frequency:", meanFrequency)
            iteration += 1
            print("--------------------------", "\n")

    # dm.soundDataManager(soundDirectory, graphDirectory, "sound_file_")
