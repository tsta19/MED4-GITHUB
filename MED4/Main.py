# imports
import glob

from DataManager import *
from Explorer import *
from NoiseReduction import *
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

    # Sample Frequency
    sampleFrequency = 44100

    # Arrays
    meanFrequencies = []

    # Class instantiations
    exp = Explorer()
    nRed = NoiseReduction()
    spectra = SpectraAnalysis()
    vr = VoiceRecognizer()

    if record:
        from Recorder import *

        rec = Recorder()
    else:
        print("Recording is turned: Off")

    if modeSingleFile:
        number = 1
        soundFile = "{}sound_file_{}.{}".format(exp.getAudioFilePath(), number, "wav")
        singleFileSpecification = os.path.join(str(soundFile))

        nRed.noiseFiltering(False, soundFile, singleFileSpecification)
        spectra.spectra("sound_file_1")
        sampleRate, data = wavfile.read(exp.getAudioFilePath() + "sound_file_1" + ".wav")
        vr.recognize(spectra.meanFrequency(data, sampleFrequency))

        if debug:
            print("Mean Frequency:", spectra.meanFrequency(data, sampleFrequency))

    if modeDirectory:
        soundFile = "mic"
        outputName = "filtered"
        for filename in glob.glob(os.path.join(exp.getAudioFilePath(), '*.wav')):
            wavFile = wave.open(filename, 'r')
            data = wavFile.readframes(wavFile.getnframes())
            wavFile.close()

        for audioFiles in glob.glob(os.path.join(exp.getAudioFilePath() + str(soundFile) + '*.wav')):
            audioFileSpecification = os.path.join(exp.getFilteredFilePath(), str(soundFile) + "_" + str(iteration) + "_"
                                                  + str(outputName) + ".wav")

            if debug:
                print("+------------------------+")
                print("For-Loop iteration:", iteration)
                print("File:", audioFileSpecification)

            nRed.noiseFiltering(True, soundFile, audioFileSpecification, audioFiles)
            spectra.spectra(audioFileSpecification)
            # sampleRate, data = wavfile.read(exp.getAudioFilePath() + soundFile + "_" + str(iteration) + "_"
            #                                 + str(outputName) + ".wav")
            samples, sampleRate = librosa.load((exp.getFilteredFilePath() + soundFile + "_" + str(iteration) + "_"
                                                + str(outputName) + ".wav"), sr=None, mono=True, offset=0.0,
                                               duration=None)

            if debug:
                print("Samples:", samples)
                print("SampleRate:", sampleRate)
                fileDuration = len(samples) / sampleRate
                print("File Duration:", round(fileDuration, 2), "second(s)")

            spectra.fastFourierTransform(samples, sampleRate)
            spectra.spectrogram(samples, sampleRate)

            # spectra.spectrogram(samples, sampleRate)

            meanFrequency = spectra.meanFrequency(samples, sampleFrequency)
            meanFrequencies.append(meanFrequency)

            if debug:
                print("Mean Frequency:", round(meanFrequency, 4))
                recognition = vr.recognize(meanFrequency)
                print("+------------------------+", "\n")

            iteration += 1
            spectra.iterator += 1

        # ----------------- WORK IN PROGRESS ---------------------------
        #     with open("Program_Summary_Data", "w") as txt:
        #         spacer = "+------------------------+"
        #         space = "\n"
        #         txt.write(spacer + space)
        #         txt.write("File: " + str(audioFileSpecification) + space)
        #         txt.write("Mean Frequencies: " + str(meanFrequencies) + space)
        #         txt.write("File Duration: " + str(len(samples)) + space)
        #         txt.write("Recognized as: " + str(recognition) + space)
        #         txt.write(spacer + space)
        #
        # txt.close()
        # ----------------- WORK IN PROGRESS ---------------------------

    # dm.soundDataManager(soundDirectory, graphDirectory, "sound_file_")