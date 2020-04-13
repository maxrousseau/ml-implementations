'''
pytorch implementation of LeNet5 1998 from the paper titled: Gradient-Based
Learning Applied to Document Recognition (LeCun et al.)

network architecture: the network consists of seven hidden lauers:

- Input: 1 channel, 32x32 image
- C1: 6 channels, 5x5 kernel -> 6 28x28 feature map
- S2: maxpool 2x2 kernel, stride 2, -> 6 14x14 feature map
- C3: 16 channels, 5x5 kernel -> 16 10x10 feature map
- S4: maxpool 2x2, stride 2 -> 16  5x5 feature map
- C5: 120 channels, 5x5 kernel -> 120 1x1 feature map
- F6: 84 units, sigmoid
- Output: 10 units, RBF

Criterion: Mean Squarred Error (I used log loss)
Optimizer: SGD
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as td
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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

def imsave(name, img):
	img = img/2+ 0.5 
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.savefig('%s%s' % ('./fig/', name))

# load data
transform = transforms.Compose([transforms.Resize((32,32)),
								transforms.ToTensor()])

trainset = td.MNIST(root='./db', train=True, download=True, transform=transform)
testset =  td.MNIST(root='./db', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True,
	num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False,
	num_workers=2)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# define network
class LeNet5(nn.Module):
	def __init__(self):
		super(LeNet5, self).__init__()
		self.c1 = nn.Conv2d(1, 6, kernel_size=(5,5))
		self.s = nn.MaxPool2d(kernel_size=(2,2), stride=2)
		self.c3 = nn.Conv2d(6, 16, kernel_size=(5,5))
		self.c5 = nn.Conv2d(16, 120, kernel_size=(5,5))
		self.f6 = nn.Linear(120, 84)
		self.f7 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.s(F.relu(self.c1(x)))
		x = self.s(F.relu(self.c3(x)))
		x = F.relu(self.c5(x))
		x = x.view(x.size(0), 120)
		x = F.relu(self.f6(x))
		x = F.log_softmax(self.f7(x), dim=-1)
		return x

net = LeNet5()

# loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training loop
info('Training started', None)
for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data

		optimizer.zero_grad()

		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if (i+1) % 1000 ==  0:
			info(('Epoch %3s | Input %6s | Running loss %6s |' %  (epoch+1, i+1,
					running_loss/i)), None)
info('Training completed', None)
PATH = './lenet5.pth'
torch.save(net.state_dict(), PATH)
info('Network saved', None)


net.load_state_dict(torch.load(PATH))
info('Network loaded', None)
# test
total = 0
correct = 0
with torch.no_grad():
	for data in testloader:
		inputs, labels = data
		outputs = net(inputs)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		correct += c.sum().item()
		total += labels.size(0)

info('Model accuracy: %3s %%' % (100*correct/total), None)
