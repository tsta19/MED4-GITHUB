# imports
import glob
import os
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
    debug = True
    record = False

    # Iterator(s)
    iteration = 1

    # Class instantiations
    exp = Explorer()

    if record:
        from Recorder import *
        rec = Recorder()

    else:
        print("Recording is turned: Off")

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
        vr.recognize(spectra.meanFrequency(data, 3000))

        if debug:
            print("Mean Frequency:", spectra.meanFrequency(data, 3000))

    if modeDirectory:
        soundFile = "noise"
        outputName = "filtered"
        for filename in glob.glob(os.path.join(exp.getAudioFilePath(), '*.wav')):
            wavFile = wave.open(filename, 'r')
            data = wavFile.readframes(wavFile.getnframes())
            wavFile.close()

        for audioFiles in glob.glob(os.path.join(exp.getAudioFilePath() + str(soundFile) + '*.wav')):
            audioFileSpecification = os.path.join(exp.getAudioFilePath(), str(soundFile) + "_" + str(iteration) + "_"
                                                  + str(outputName) + ".wav")

            if debug:
                print("--------------------------")
                print("For-Loop iteration:", iteration)
                print("File:", audioFileSpecification)

            nRed.noiseFiltering(True, soundFile, audioFileSpecification, audioFiles)
            spectra.spectra(audioFileSpecification)
            sampleRate, data = wavfile.read(exp.getAudioFilePath() + soundFile + "_" + str(iteration) + "_"
                                            + str(outputName) + ".wav")

            meanFrequency = spectra.meanFrequency(data, 5000)
            vr.recognize(meanFrequency)

            if debug:
                print("Mean Frequency:", meanFrequency)
                print("--------------------------", "\n")

            iteration += 1
            spectra.iterator += 1

    # dm.soundDataManager(soundDirectory, graphDirectory, "sound_file_")
