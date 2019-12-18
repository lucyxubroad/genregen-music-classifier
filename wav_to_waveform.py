import torch
import torchaudio
import matplotlib.pyplot as plt
import os

genreDict = {}

for subdirs, dirs, files in os.walk('genres'):
    for dirName in dirs:
        genreDict[str(dirName)] = []
        currDir = os.path.abspath(os.path.join('genres', str(dirName)))
        for wavFile in os.listdir(currDir):
            if wavFile[-4:] == ".wav":
                print("Reading " + wavFile + " now...\n")
                waveform, sample_rate = torchaudio.load(os.path.join(currDir, wavFile))

                fig = plt.figure()
                plt.plot(waveform.t().numpy())
                figLoc = os.path.join('waveforms', wavFile[:wavFile.rfind('.')] + '.png')
                fig.savefig(figLoc)
                genreDict[str(dirName)].append(wavFile[:wavFile.rfind('.')])

print(genreDict)