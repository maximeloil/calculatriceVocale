import os
import skimage.io
import skimage.transform
from torch.utils.data import Dataset, DataLoader
from numpy import transpose as trsp
from torch import from_numpy as np2Tens
import numpy
import sys
sys.path.append("C://Users//space_000//Documents//Ecole//Polytechnique//2ème année//MODAL//Emilien&Maxime//python_src")
import loadData

class spectrogramDataset(Dataset):
    """Image spectrograms dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        if 'Thumbs.db' in os.listdir(self.root_dir):
            return len(os.listdir(self.root_dir))-1
        else:
            return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir,str(idx%14)+'-'+str((idx//14)+1)+'.wav')
        image = loadData.getSpectrogramFromPath(wav_name)
        image = image.astype(numpy.float64)
        image = np2Tens(image)
        image = image.double()
        image = image.reshape((1,256,256))
        image = image/(abs(image).max())
        label = idx%14
        return image,label
  