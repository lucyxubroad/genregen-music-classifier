import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from classifiers import Net
import torch.nn as nn

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.ImageFolder('../test_waveforms', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True, num_workers=2)

classes = [
    ('pop', 'reggae', 'country', 'hiphop', 'classical', 'blues', 'metal', 'rock', 'jazz', 'disco'),
    ('pop', 'country', 'hiphop', 'classical', 'blues', 'rock', 'jazz'),
    ('country', 'hiphop/pop', 'classical', 'rock', 'jazz') # reflected in training waveform folder
]

PATH = './nets/neural_net_v1.pth'
training_classes = classes[2]
num_classes = len(training_classes)
dataiter = iter(testloader)
images, labels = dataiter.next()

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

print(outputs)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % training_classes[predicted[j]] for j in range(num_classes)))

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

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
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

for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (training_classes[i], 100 * class_correct[i] / class_total[i]))

