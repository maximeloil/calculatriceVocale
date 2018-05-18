from dataLoader import createDataLoader
import argparse
from ipdb import set_trace as st

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='C:/Users/eleves/Documents/Emilien&Maxime/data/wav/', help='Path to the data directory')
parser.add_argument('--window_size', type=int, default=65000)
parser.add_argument('--step_value', type=int, default=13000)
opt = parser.parse_args()

trainLoader = createDataLoader(opt)
lenTrainLoader = len(trainLoader)

nEpoch = 1

for epoch in range(nEpoch):
    trainIter=iter(trainLoader)
    for it, (label, logspect) in enumerate(trainLoader):
        # label,spect va prendre les valeurs w_class, spectrogram
        # enumerate prend l'indice du trainLoader 
        i = 0
        # st()
