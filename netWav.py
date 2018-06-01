import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io, transform
import sys
sys.path.append("C://Users//space_000//Documents//Ecole//Polytechnique//2ème année//MODAL//Emilien&Maxime//python_src")
from createDataLoaderWav import spectrogramDataset


path = "C://Users//space_000//Documents//Ecole//Polytechnique//2ème année//MODAL//data//wav//"


##Data collecting + def of the classes

spectr_dataset_train = spectrogramDataset(root_dir=path+'train//')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                    
trainloader = torch.utils.data.DataLoader(spectr_dataset_train, batch_size=4,
                        shuffle=True, num_workers=4)
                    
                 
spectr_dataset_test = spectrogramDataset(root_dir=path+'test//')
testloader = torch.utils.data.DataLoader(spectr_dataset_test, batch_size=4,
                        shuffle=True, num_workers=4)

classes = ('zero', 'un', 'deux', 'trois',
           'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf', 'plus','fois','moins','divise_par')
           
           

    
##test: print some images
'''
# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
plt.show()
'''


##def of the Neurol Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.fc1 = nn.Linear(4 * 61 * 61, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 14)

    def forward(self, x):
        #x.shape = [4, 1, 256, 256]
        x=torch.FloatTensor(x.numpy())
        x = self.pool(F.relu(self.conv1(x)))
        #x.shape = [4, 2, 126, 126]
        x = self.pool(F.relu(self.conv2(x)))
        #x.shape = [4, 4, 61, 61]
        x = x.view(-1, 4 * 61 * 61)
        #x.shape = [4*61*61]
        x = F.relu(self.fc1(x))
        #x.shape = [256]
        x = F.relu(self.fc2(x))
        #x.shape = [128]
        x = self.fc3(x)
        #x.shape = [14]
        return x


net = Net()


##def of the loss function

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



##Training!

for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')


##Test the net on test images

dataiter = iter(testloader)
images, labels = dataiter.next()


outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the '+str(total)+' test images: %d %%' % (
    100 * correct / total))
    
    
    
class_correct = list(0. for i in range(14))
class_total = list(0. for i in range(14))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(14):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
        


##To compute with GPU for speedup (not avalaible on my computer)
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)

net.to(device)
inputs, labels = inputs.to(device), labels.to(device)
'''