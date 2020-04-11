#!/usr/bin/env python
# coding: utf-8
'''
Training a Classifier
--------------------

Loading data: use third party libraries to convert data into numpy arrays.
You can then convert the numpy array.
- images (Pillow, Opencv)
- audio (scipy, librosa)
- text (std library, NLTK, SpaCy)
torchvision has datasets, tranforms and dataloader (basically avoids boilerplate code)

this tutorial uses CIFAR10

steps:
1. load and normalize the CIFAR10 training and test sets
2. defube a CBB
3. define a loss fcn
4. train net on training set
5. test the net on test set
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import logging
import pprint

def info(MSG, VALUE):
	logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
	pp = pprint.PrettyPrinter(indent=4)
	logging.info(MSG)
	if VALUE != None:
		pp.pprint(VALUE)
	else:
		None
	print('-'*80)

# here we look at how we can display the data
def imsave(name, img):
    img = img/2+ 0.5 # this will unnormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig('%s%s' % ('./fig/', name))
'''
The compose function will chain together image transformations (matter of convenience)
ToTensor() will convert the array to a pytorch tensor
Normalize will normalize the image to allow for faster convergence
-> in this case you input mean and std dev for each channel (R, G, B)
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# here we load the data and apply the transform defined above
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# DataLoader will create iterable batches from the dataset provided
# it also has other functionalities (i.e. shuffle)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# get random images from the dataset
dataiter = iter(trainloader) # iter creates an iterable object from list
images, labels = dataiter.next() # .next call will return the element of the array
                                # in this case it is the element at 0
                                # batch is 4 (see above) so we have 4 images and 4 labels
# show the image -> make_grid
imsave('1-sample', torchvision.utils.make_grid(images))
# print labels
# % = substitue, 5 = 5 space padding, s = string type
info(('Classes fig 1:' + ' '.join('%5s' % classes[labels[j]] for j in range(4))),
 None)

# 2 here we create our network in this case this is a CNN
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        '''
        Here you basically define the various affine functions that you will
        use for your network

        Time for a quick recap on convolutions
        - the concept is basically a filter (i.e. kernel = 5x5) that is applied to an
          image (i.e input = 10x10) by sliding along it 
          (i.e. stride = 1 -> output/feature map 6x6)
        - another way to think about it is the sliding dot product of the image
        - convolutions are usually implemented as cross-correlation in ML libraries
          (we begin at the top left corner)

        see: https://cs231n.github.io/convolutional-networks/

        **idea: write library to automatically verify matrix dimensions**

        NOTE: CIFAR10 -> 32x32 images
        Conv2D (in_channels, out_channels, kernel_size)
        since the image is RGB we know that the
        the out channel parameter is basically the number of filters you
        want to use

        to compute the results from the conv on the output use this
        formulae:
            W(input volume), F(receptive field size), S(stride), P(padding)
            ((W - F +2P)/S)+1
            (32 - 5 + 2*0)/1) + 1 = 28x28

        to compute the result from the pool layer use this formulae:
            F(spatial extent), S(stride)
            ((W-F)/S)+1
            ((28-2)/2)+1 = 14x14

        32x32x3 (conv1)-> 28x28x8
        28x28x8 (maxpool)-> 14x14x8
        14x14x8 (conv2)-> 10x10x16
        10x10x16 (maxpool)-> 5x5x16
        '''
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''
        here is where you assemble the affine and the activation
        function for each layer of the network
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) #flatten the matrix to vector for the fc layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# 3 here we define loss function and our weight update function
import torch.optim as optim
'''
Criterion: is our loss function, in this case we use pytorch's implementation
of cross entropy loss (also know as log loss).

	It is a convex loss function, see explanation here:
	(https://www.youtube.com/watch?v=HIQlmHxI6-0)

Optimizer: is the function we use to update weights. Here we use pytorch's
implementation of SGD (stochastic gradient descent).
'''
criterion = nn.CrossEntropyLoss() # idk what this is***
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4 training the network, we loop over the data iterator
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get inputs: data is list of [inputs, labels]
        inputs, labels = data

        # zero the parameter fradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini batches
            info(('[%d, %5d] loss: %.3f' % 
                     (epoch + 1, i + 1, running_loss / 2000)), None)
            running_loss = 0.0

print('Finished Training')

# save the trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# 5 test the network on the test data
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images to see what we're working with
imsave('2-image_processed', torchvision.utils.make_grid(images))
info('GroundTruth of figure 2: ', ' '.join('%5s' % classes[labels[j]] for j in
range(4)), None)


'''
it is not necessary to do so in this case but for demonstration
purposes we load the model that we saved
'''
net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)

# what is this??***
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

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
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


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

'''
* Training on gpu
define your device
'''
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Assuming that we are on a CUDA machine, this should print a CUDA device:
# print(device)
