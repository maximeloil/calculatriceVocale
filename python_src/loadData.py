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

def writeWavFromArray(sample, namefile):
    Fe = 44100
    scaled = np.int16(sample/np.max(np.abs(sample)) * 32767) #coding on 16 bits
    wavfile.write(namefile+".wav", Fe, scaled)

def addNoiseFromPath(path):
    rate,sample = wavfile.read(path)
    if type(sample[0])==np.ndarray:
        sample = sample[:,0]
    levelOfNoise = 0.1*np.average(abs(sample))
    noise = np.random.normal(0,levelOfNoise,len(sample))
    sample += noise
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

#TODO change tempo https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

#TODO change gain https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

#TODO change high

def createSpectrogramFromPath(path):
    rate, sample = wavfile.read(path)
    if type(sample[0])==np.ndarray:
        sample = sample[:,0]        
    Fe = 44100
    f, t, Sxx = signal.spectrogram(sample, Fe,nfft=511,nperseg=len(sample)//225)
    
    Sxx = np.resize(Sxx, (256,256)) 
    f = np.resize(f,256)
    t = np.resize(t,256)
    print(Sxx.shape)
    norm = cls.Normalize(vmin=-1.,vmax=1.)
    norm = cls.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    st()
    img = plt.pcolormesh(t, f, Sxx,norm=norm,cmap='jet')

    return img

def getClassFromString(filepath):
    fileBasename = os.path.basename(filepath)
    return fileBasename.split('-')[0]

class DatasetFromFolder(data.Dataset):
    def __init__(self):
        print('Initializing dataset from folder')
        # fix
        self.filesList = [os.path.join(opt.path, filename) for filename in sorted(os.listdir(opt.path))][1:]

    def __len__(self):
        return len(self.filesList)

    def __getitem__(self, index):
        # for each file in x:
        #   get class from string
        #   load spectrogram
        #   return class, spectrogram
        filepath = self.filesList[index]
        print(filepath)
        w_class = getClassFromString(filepath)
        spectrogram = createSpectrogramFromPath(filepath)
        return w_class, spectrogram

    
