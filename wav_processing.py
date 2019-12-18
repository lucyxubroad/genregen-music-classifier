import torch
import torchaudio
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


mpl.rcParams['agg.path.chunksize'] = 10000

for subdirs, dirs, files in os.walk('genres'):
    for dirName in dirs:
        print("Starting " + str(dirName))
        if not os.path.exists(os.path.join('waveforms', str(dirName))):
            os.mkdir(os.path.join('waveforms', str(dirName)))
        currDir = os.path.abspath(os.path.join('genres', str(dirName)))
        for audioFile in os.listdir(currDir):
            if audioFile[-4:] == ".wav" or audioFile[-4:] == ".mp3":
                waveform, sample_rate = torchaudio.load(os.path.join(currDir, audioFile))
                fig = plt.figure(figsize=(15, 4), dpi=100)
                plt.plot(waveform.t().numpy())
                
                plt.axis('off')
                axes = plt.gca()
                axes.set_ylim([-1.25, 1.25])
                figLoc = os.path.join('waveforms', dirName, audioFile[:audioFile.rfind('.')] + '.png')
                fig.savefig(figLoc, bbox_inches="tight")
                plt.close(fig)
        print("Done with " + str(dirName))