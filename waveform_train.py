import torch
import torchvision

# common image transformations, can be chained together using Compose
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision as tv
import pandas as pd

from torchvision.models import resnet18
from torchvision.models import resnet34

# 1. Loading and normalizing waveforms

# Apply a list of transformations (in this case, just normalization)
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),   # normalize a tensor image with mean and standard deviation
                                                                    # normalize each channel of the input torch.*Tensor 
                            (0.5, 0.5, 0.5))])

print("downloading train set")
# Download training data set from TorchVision
trainset = torchvision.datasets.ImageFolder('../waveforms6', transform=transform)

# Download testing data set from TorchVision
print("downloading test set")
testset = torchvision.datasets.ImageFolder('../test_waveforms6', transform=transform)
print("downloading train loader")
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True, num_workers=2)

# classes = ('pop', 'reggae', 'country', 'hiphop', 'classical', 'blues', 'metal', 'rock', 'jazz', 'disco')
# classes = ('pop', 'country', 'hiphop', 'classical', 'blues', 'rock', 'jazz')
classes = ('country', 'hiphop_pop', 'classical', 'rock', 'jazz')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 2. Define a Convolution Neural Network
class Net(nn.Module): # What is all of this?
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 6) # Convolution Layer 1
        # in_channels (depth of input image), out_channels (desired depth of output), kernel size (size of convolution layer)
        self.pool = nn.MaxPool2d(2, 2) # maxpooling Layer
        self.conv2 = nn.Conv2d(3, 16, 5) # Convolution Layer 2
        self.fc1 = nn.Linear(364416, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 364416)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# print("Net()")
# net = Net()
 

# 2b. Defining A Residual Neural Network

class RNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        #x = self.model.layer4(x)
        x = x.view(-1, 512)
        # x = x.view(-1, x.size(0))
        x = self.model.fc(x)

        return x

net = tv.models.resnet18()
# net = RNet(model=resnet18(pretrained=False)) # comment out this line to use Net()
# model=resnet18(pretrained=False)

# 3. Define a loss function and optimizer

print("defining a loss function and optimizer")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. Train the network - loop over data iterator, feeding inputs to network, optimize 

print("into training...")
for epoch in range(20):  # loop over the dataset multiple times
    print("Downloading training data...")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=2)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
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
                  (epoch + 1, i + 1, running_loss/10))
            running_loss = 0.0

print('Finished Training')

PATH = './genregen_20e5gsResNet.pth'
torch.save(net.state_dict(), PATH)

# # 5. Test CNN on test data

dataiter = iter(testloader)
images, labels = dataiter.next()
print(labels)

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(5)))

# net = Net()
net = tv.models.resnet18()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

print(outputs)
_, predicted = torch.max(outputs, 1)
print(predicted)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(5)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

