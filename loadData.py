from scipy import signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import wave
from ipdb import set_trace as st
import os
import torch.utils.data as data
import numpy as np
import random

def writeWavFromArray(sample, namefile):
    Fe = 44100
    scaled = np.int16(sample/np.max(np.abs(sample)) * 32767) #coding on 16 bits
    wavfile.write(namefile+".wav", Fe, scaled)

def addNoise(sample, Fe):
   
    #perNoise = np.random.normal(0,levelOfNoise,len(sample))
    levelOfNoise = 0.1*np.average(abs(sample))
    lenght = len(sample)
    print('LevelOfNoise')
    print(levelOfNoise)
    noise = np.random.normal(0,levelOfNoise,lenght)
    #noise.astype(int)
    print('Sample')
    print(sample)
    print('Noise')
    print(noise)
    #SampleNoise = sample + noise #ne marche pas ?
    wavfile.write("C:/Users/eleves/Documents/Emilien&Maxime/test_python/SamplePlusNoise.wav", Fe, sample + noise)
    print('Sample + Noise')
    return (sample + noise)

#TODO change tempo https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

#TODO change gain https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py

#TODO change high

def crop_sample(sample, window_size=65000, step_value=13000):
    max_mean=0
    for i in range(sample.size//step_value):
        start = i*step_value
        end = start + window_size

        mean=np.abs(sample[start:end]).mean() if end <= sample.size else np.abs(sample[(sample.size-window_size):sample.size]).mean()
        
        if mean > max_mean:
            max_mean=mean
            if end > sample.size:
                best_cropped_sample=sample[(sample.size-window_size):sample.size]
            else:
                best_cropped_sample=sample[start:end]
    
    return best_cropped_sample

def fill_sample(sample, window_size):
    zero_sample=np.zeros(window_size)
    zero_sample[:sample.size] = sample
    return zero_sample

def createSpectrogramFromPath(path, window_size, step_value):

    rate, sample = wavfile.read(path)
    if type(sample[0])==np.ndarray:
        sample = sample[:,0]        
    Fe = 44100

    # Randomization
    #sample = changePitch(sample)
    #sample = changeSpeed(sample)
    sample = addNoise(sample, rate)

    if sample.size > window_size:
        sample = crop_sample(sample, window_size, step_value)
    else:
        sample=fill_sample(sample, window_size)

    f, t, Sxx = signal.spectrogram(sample, Fe,nfft=511,nperseg=len(sample)//225)
    Sxx = Sxx[:,:256]
    # Sxx = np.resize(Sxx, (256,256)) 
    # f = np.resize(f,256)
    # t = np.resize(t,256)
    # print(Sxx.shape)
    # norm = cls.Normalize(vmin=-1.,vmax=1.)
    # norm = cls.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    # img = plt.pcolormesh(t, f, Sxx,norm=norm,cmap='jet')
    return np.log1p(Sxx)

def getClassFromString(filepath):
    fileBasename = os.path.basename(filepath)
    return fileBasename.split('-')[0]

class DatasetFromFolder(data.Dataset):
    def __init__(self, opt):
        print('Initializing dataset from folder')
        self.opt = opt
        self.filesList = [os.path.join(opt.path, filename) for filename in sorted(os.listdir(opt.path))]

    def __len__(self):
        return len(self.filesList)

    def __getitem__(self, index):
        # for each file in x:
        #   get class from string
        #   load spectrogram
        #   return class, spectrogram
        filepath = self.filesList[index]
        w_class = getClassFromString(filepath)
        spectrogram = createSpectrogramFromPath(filepath, window_size=self.opt.window_size, step_value=self.opt.step_value)
        return w_class, spectrogram

    
