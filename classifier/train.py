import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from classifiers import Net
import torch.nn as nn

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print("Downloading train set.")
trainset = torchvision.datasets.ImageFolder('../waveforms', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=2)

epochs = [1, 3, 5, 10, 15, 20]
learning_rates = [0.001, 0.02, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
momentums = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
classes = [
    ('pop', 'reggae', 'country', 'hiphop', 'classical', 'blues', 'metal', 'rock', 'jazz', 'disco'),
    ('pop', 'country', 'hiphop', 'classical', 'blues', 'rock', 'jazz'),
    ('country', 'hiphop/pop', 'classical', 'rock', 'jazz') # reflected in training waveform folder
]

print("Creating neural net.")
net = Net()

criterion = nn.CrossEntropyLoss()
parameters = net.parameters()
learning_rate = learning_rates[0]
momentum = momentums[8]

optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum)

epoch_num = epochs[2]


print("Beginning data training.")
running_loss = 0.0
for epoch in range(epoch_num):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data # get the inputs; data is a list of [inputs, labels]
        optimizer.zero_grad() # zero the parameter gradients
        # forward, backward, optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics of training
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/10))
            running_loss = 0.0

print("Finishing training.")

PATH = './nets/neural_net_v1.pth'
torch.save(net.state_dict(), PATH)