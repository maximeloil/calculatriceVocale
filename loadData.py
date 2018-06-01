from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
import scipy.io.wavfile as wavfile
import matplotlib.colors as cls
import wave
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pydub import AudioSegment

#========================================================================================#
##Manipulating data

absolutePath = "C://Users//space_000//Documents//Ecole//Polytechnique//2ème année//MODAL//data//wav//"

def listenArray(sample,Fe=44100):
    sampleCopy = np.copy(sample)
    pygame.mixer.pre_init(Fe, size=-16, channels=1)
    pygame.mixer.init()
    if type(sampleCopy[0])==np.ndarray:
        sampleCopy = sampleCopy[:,0]
    sampleCopy = sampleCopy.copy(order='C')
    sampleCopy = np.int16(sampleCopy/np.max(np.abs(sampleCopy)) * 32767) #codes on 16 bits
    sound = pygame.sndarray.make_sound(sampleCopy)
    sound.play()

def writeWavFromArray(sample, namefile, Fe=44100):
    scaled = np.int16(sample/np.max(np.abs(sample)) * 32767) #codes on 16 bits
    wavfile.write(namefile+".wav", Fe, scaled)
    

#========================================================================================#
##Transfom functions


def prepSample(sample):
    sampleCopy = np.copy(sample)
    sampleCopy.setflags(write=True)
    #keeps only left speaker
    if type(sampleCopy[0])==np.ndarray:
        sampleCopy = sampleCopy[:,0]
    return sampleCopy
    
    
#still understandable for rateOfNoise = 2
def addNoiseFromPath(path, rateOfNoise=0.1,Fe = 44100):
    rate,sample = wavfile.read(path)
    #keeps only left speaker
    if type(sample[0])==np.ndarray:
        sample = sample[:,0]
    levelOfNoise = rateOfNoise*np.average(abs(sample))
    noise = np.random.normal(0,levelOfNoise,len(sample))
    sample += noise
    return sample
    
def addNoiseFromSample(sample, rateOfNoise=0.1,Fe = 44100):
    sampleCopy = prepSample(sample)
    levelOfNoise = rateOfNoise*np.average(abs(sampleCopy))
    noise = np.random.normal(0,levelOfNoise,len(sampleCopy))
    noise = noise.astype(np.int16)
    sampleCopy += noise
    return sampleCopy

#requires scale > 1
def rescaleArray(sample,scale):
    sampleCopy = prepSample(sample)
    r = np.zeros(int(len(sampleCopy)*scale))
    for i in range(0,len(r)-1):
        r[i] = (np.ceil(i/scale)-i/scale)*sampleCopy[int(np.floor(i/scale))]+(1-(np.ceil(i/scale)-i/scale))*sampleCopy[int(np.ceil(i/scale))]
    r[-1]=sampleCopy[-1]
    return r


def changeHigh(sample,scale):
    sampleCopy = prepSample(sample)
    n = len(sampleCopy)
    N = int(2**np.ceil(np.log2(n)))
    f = scipy.fftpack.rfft(sampleCopy,N)
    
    if scale > 1:
        y = np.zeros(int(2**np.ceil(np.log2(len(f)*scale))))
        r=rescaleArray(f,scale)
        y[:len(r)]=r
    else:
        y = np.zeros(len(f))
        r=rescaleArray(f,scale)
        y[:len(r)]=r
    
    
    r = scipy.fftpack.irfft(y)
    return r[:n]
    
def changeHigh2(sample,freqOffSet, Fe = 44100):
    sampleCopy = prepSample(sample)
    n = len(sampleCopy)
    N = int(2**np.ceil(np.log2(n)))
    f = scipy.fftpack.rfft(sampleCopy,N)
    
    y=np.zeros(len(f))
    if freqOffSet >= 0 :
        for i in range(1,len(f)):
            if i+int(len(f)*freqOffSet/Fe) >= len(y):
                break
            else:
                y[i+int(len(f)*freqOffSet/Fe)] = f[i]
    else:
        for i in range(len(f)-1,-1,-1):
            if i+int(len(f)*freqOffSet/Fe) < 0:
                break
            else:
                y[i+int(len(f)*freqOffSet/Fe)] = f[i]
        
    
    
    r = scipy.fftpack.irfft(y)
    return r[:n]
    

#TODO change tempo https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py


#TODO change high

#zero padding after a sample to reach a size of window_size
def fill_sample(sample, window_size=65000):
    zero_sample=np.zeros(window_size)
    zero_sample[:sample.size] = sample
    return zero_sample
    
#crops a sample such as the result has a size of window_size
def crop_sample(sample, window_size=65000, step_value=13000):
    max_mean=0
    sampleCopy = np.copy(sample)
    #keeps only left speaker
    if type(sampleCopy[0])==np.ndarray:
        sampleCopy = sampleCopy[:,0]
    if sampleCopy.size < window_size:
        return fill_sample(sampleCopy,window_size)
    for i in range(sampleCopy.size//step_value):
        start = i*step_value
        end = start + window_size
        if end <= sampleCopy.size :
            mean = np.abs(sampleCopy[start:end]).mean()
        else:
            mean = np.abs(sampleCopy[(sampleCopy.size-window_size):sampleCopy.size]).mean()
        if mean > max_mean:
            max_mean=mean
            if end > sampleCopy.size:
                best_cropped_sample=sampleCopy[(sampleCopy.size-window_size):sampleCopy.size]
            else:
                best_cropped_sample=sampleCopy[start:end]
    return best_cropped_sample
    
#determines the start and the stop of the instruction
def findStartAndStop(sample,Fe=44100):
    sampleCopy = np.copy(sample)
    #keeps only left speaker
    if type(sampleCopy[0])==np.ndarray:
        sampleCopy = sampleCopy[:,0]
    ws =  Fe//50 #size of a window of 20ms
    ws2 = ws//2
    #sample = abs(sample)
    n = sampleCopy.size
    movAvg = np.zeros(n)
    for i in range(0,n):
        movAvg[i] = np.average(sampleCopy[max(i-ws2,0):min(i+ws2+1,n)])
    return movAvg



#========================================================================================#
##Preprocessing functions

#create a spectrogram (log scale for colours)
def createSpectrogramFromPath(path, window_size=65000, step_value=13000):
    rate, sample = wavfile.read(path)
    sampleCopy = np.copy(sample)
    if type(sampleCopy[0])==np.ndarray:
        sampleCopy = sampleCopy[:,0]        
    Fe = 44100
    if sampleCopy.size > window_size:
        sampleCopy = crop_sample(sampleCopy, window_size, step_value)
    #else:
        #sampleCopy=fill_sample(sampleCopy, window_size)
    f, t, Sxx = signal.spectrogram(sampleCopy, Fe,nfft=511,nperseg=len(sampleCopy)//225)
    f = f[:256]
    t = t[:256]
    Sxx = Sxx[:,:256]
    Sxx = abs(Sxx)
    fillWithMin(Sxx)
    figure = plt.figure(  )
    plot   = figure.add_subplot ( 111 )
    norm = cls.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    img = plt.pcolormesh(t, f, Sxx,norm=norm,cmap='jet')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    figure.savefig("C://Users//space_000//Documents//Ecole//Polytechnique//2ème année//MODAL//data//img//train//"+os.path.basename(path).replace('.wav','.png'),bbox_inches=0)

#create a spectrogram (log scale for colours)
def getSpectrogramFromPath(path, window_size=65000, step_value=13000):
    rate, sample = wavfile.read(path)
    sampleCopy = np.copy(sample)
    if type(sampleCopy[0])==np.ndarray:
        sampleCopy = sampleCopy[:,0]        
    Fe = 44100
    if sampleCopy.size > window_size:
        sampleCopy = crop_sample(sampleCopy, window_size, step_value)
    #else:
        #sampleCopy=fill_sample(sampleCopy, window_size)
    f, t, Sxx = signal.spectrogram(sampleCopy, Fe,nfft=512,nperseg=len(sampleCopy)//224)
    f = f[:256]
    t = t[:256]
    tmp = Sxx[:256,:256]
    Sxx = np.zeros((256,256))
    l = len(tmp[0])
    Sxx[:,:l]=tmp
    Sxx = abs(Sxx)
    fillWithMin(Sxx)
    logSxx = np.log1p(Sxx)
    return logSxx.reshape((1,256,256))
    
#replaces the 0 of sample by the min value > 0
def fillWithMin(sample):
    max = sample.max()
    sample[sample==0.0] = max+1
    min = sample.min()
    sample[sample==max+1] = min
'''
#create a spectrogram (log scale for colours)
def createSpectrogramFromSample(sample, window_size, step_value):
    sampleCopy = np.copy(sample)
    if type(sampleCopy[0])==np.ndarray:
        sampleCopy = sampleCopy[:,0]        
    Fe = 44100
    if sampleCopy.size > window_size:
        sampleCopy = crop_sample(sampleCopy, window_size, step_value)
    else:
        sampleCopy=fill_sample(sampleCopy, window_size)
    f, t, Sxx = signal.spectrogram(sampleCopy, Fe,nfft=511,nperseg=len(sampleCopy)//225)
    f = f[:256]
    t = t[:256]
    Sxx = Sxx[:,:256]
    Sxx = abs(Sxx)
    fillWithMin(Sxx)
    figure = plt.figure(  )
    plot   = figure.add_subplot ( 111 )
    norm = cls.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    img = plt.pcolormesh(t, f, Sxx,norm=norm,cmap='jet')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    f.savefig("C://Users//space_000//Documents//Ecole//Polytechnique//2ème année//MODAL//data//img//test2.png",bbox_inches=0)
'''
def getClassFromString(filepath):
    fileBasename = os.path.basename(filepath)
    return fileBasename.split('-')[0]

class DatasetFromFolder(Dataset):
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


for i in range(0,14):
    for j in range(1,36):
        print(str(i)+"-"+str(j))
        createSpectrogramFromPath(absolutePath+"train//"+str(i)+"-"+str(j)+".wav")

'''
Y=[]
for x in os.listdir(absolutePath+'test//'):
    if x[len(x)-4:len(x)]=='.wav':
        rate,sample=wavfile.read(absolutePath+'test//'+x)
        Y.append(abs(sample).max())
for x in os.listdir(absolutePath+'train//'):
    if x[len(x)-4:len(x)]=='.wav':
        rate,sample=wavfile.read(absolutePath+'train//'+x)
        Y.append(abs(sample).max())
Y=np.array(Y)
print(Y.max())
'''