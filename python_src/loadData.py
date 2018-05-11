from scipy import signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import wave
from ipdb import set_trace as st
import os
import argparse
import torch.utils.data as data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='C:/Users/eleves/Documents/Emilien&Maxime/data/wav/', help='Path to the data directory')
opt = parser.parse_args()

#TODO add noise https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

#TODO change tempo https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

#TODO change gain https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

#TODO change high

def createSpectrogramFromPath(path):
    rate,sample = wavfile.read(path)
    if type(sample[0])==np.ndarray:
        sample = sample[:,0]
    Fe = 44100
    f, t, Sxx = signal.spectrogram(sample, Fe,nfft=511,nperseg=len(sample)//225)
    st()
    Sxx = np.resize(Sxx, (256,256))
    f = np.resize(f,256)
    t = np.resize(t,256)
    print(Sxx.shape)
    norm = cls.Normalize(vmin=-1.,vmax=1.)
    norm = cls.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    img = plt.pcolormesh(t, f, Sxx,norm=norm,cmap='jet')
    return img

def getClassFromString(filepath):
    fileBasename = os.path.basename(filepath)
    return fileBasename.split('-')[0]

class DatasetFromFolder(data.Dataset):
    def __init__(self):
        print('Initializing dataset from folder')
        self.filesList = sorted(os.listdir(opt.path))

    def __len__(self):
        return len(self.filesList)

    def __getitem__(self, index):
        # for each file in x:
        #   get class from string
        #   load spectrogram
        #   return class, spectrogram
        filepath = self.filesList[index]
        w_class = getClassFromString(filepath)
        spectrogram = createSpectrogramFromPath(filepath)
        return w_class, spectrogram

    
