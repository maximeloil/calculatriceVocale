import torch
from loadData import DatasetFromFolder

def createDataLoader(opt):
    setDataLoader=DatasetFromFolder(opt)
    dataLoader=torch.utils.data.DataLoader(setDataLoader,batch_size=1, shuffle=True, num_workers=2)
    return dataLoader