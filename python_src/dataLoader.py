import torch
from loadData import DatasetFromFolder

def createDataLoader():
    setDataLoader=DatasetFromFolder()
    dataLoader=torch.utils.data.DataLoader(setDataLoader,batch_size=1, shuffle=True, num_workers=2)
    return dataLoader