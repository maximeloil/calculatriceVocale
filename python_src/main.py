from dataLoader import createDataLoader
from ipdb import set_trace as st

trainLoader = createDataLoader()
lenTrainLoader = len(trainLoader)

nEpoch = 1

for epoch in range(nEpoch):
    trainIter=iter(trainLoader)
    for it, (label, spect) in enumerate(trainLoader):
        # label,spect va prendre les valeurs w_class, spectrogram
        # enumerate prend l'indice du trainLoader 
        st()

