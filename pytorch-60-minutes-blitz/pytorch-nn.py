#!/usr/bin/env python
# coding: utf-8
'''
Neural Networks
---------------

Constructed using ```torch.nn```, ```nn``` depends on ```autograd``` to define
and differentiate the models. ```nn.Module``` contain layers and
```forward(input)``` method returning ```output```.

Procedure to follow:

1. define neural network with weights (learnable parameters)
2. iterate over dataset (inputs)
3. process input through network
4. compute loss
5. backpropagation of gradients using the loss
6. update weights (with an update rule: ```weight = weight - learning_rate * gradient```)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import pprint

def info(MSG, VALUE):
	logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
	pp = pprint.PrettyPrinter(indent=4)
	logging.info(MSG)
	pp.pprint(VALUE)
	print('-'*80)

# 1. define a network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels,
        # 3x3 square convolution kernel
        # nn.Conv2d docs: https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # affine operation: y = Wx + b
        # nn.Linear function from docs:
        #  ->  Applies a linear transformation to the incoming data: y = xA^T + by
        self.fc1 = nn.Linear(16*6*6, 120) # 6*6 img dimensions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # max pool: https://computersciencewiki.org/index.php/Max-pooling_/_Pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # If the size is a square, specify only one number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) # we flatten the matrix for the fully connected layers, -1 infers the correct number
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # gives us all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
info('we create a network as a class', net)
'''
Notes on the creation of a network

You need to manually define the ```forward``` function, however the
```backward``` function is automatically defined by using ```autograd```.  The
learnable parameters are returned in the ```net.parameters()```
'''
params = list(net.parameters())
info('listing the parameters', len(params))
# conv1's weights dimensions -> kernel
info('size of the parameters, in this case a kernel', params[0].size())

# 32x32 input MNIST size
input = torch.randn(1,1,32,32)
out = net(input)
info('dummy input used to create a network object', out)
# we zero the gradients and give them dummy values for a dummy backprop
net.zero_grad()
out.backward(torch.randn(1, 10))
'''
documentation says torch.nn only support mini-batches of input not a single
sample.

example: nn.Conv2d takes a 4D tensor ->  nSamples x nChannels x Heights x Width

- sample is the size of the mini-batch
- channel is the color depth (i.e. 3 for RGB)
- height and width are the dimensions of the image

loss function, takes both the (output, target) the then computes how
far the output is from the target.
there are built-in loss functions such nn.MSELoss (which stands for mean
squared error loss) -> https://en.wikipedia.org/wiki/Mean_squared_error
'''
output = net(input) # dummy input from above
target = torch.randn(10) # dummy target
target = target.view(1,-1) # = same shape at the output
# criterion is the loss function that we will use to compare our output/target
criterion = nn.MSELoss()

loss = criterion(output, target)
info('this is our loss', loss)
info('this is our target', target)
info('this is the gradient function', loss.grad_fn) # mse loss
info('going one step back in the gradients', loss.grad_fn.next_functions[0][0]) # linear
info('going two steps back in the gradients', loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

'''
Backpropagation: to propagate the computed error we call loss.backward()
NOTE: you need to CLEAR the existing gradients beforehand (otherwise they will be accumulated)
'''
net.zero_grad() #zero the gradient buffer for all parameters

info('conv1.bias.grad before backward')
info(net.conv1.bias.grad)

loss.backward()

info('conv1.bias.grad after backward')
info(net.conv1.bias.grad) # we can see that this is a vector
info(net.conv1.weight.grad) # we can see that this is a matrix
# Updating the weights: a simple update rule is SGD (stochastic gradient descent)
# -> weight = weight - learning_rate * gradient
# in this case we are doing this manually
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)

'''
torch.optim has implementation of methods such as SGD, Adam ans RMSProp

REVIEW: the optimizer is simply the function that is used to update the
weights of the various function with respect to the computed gradient.
'''
import torch.optim as optim
# create the optimizer, in this case SGD
optimizer = optim.SGD(net.parameters(), lr=0.1)
# in trainning loop
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # this does the update
