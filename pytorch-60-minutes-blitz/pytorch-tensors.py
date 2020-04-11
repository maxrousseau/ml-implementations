#!/usr/bin/env python
# coding: utf-8
'''
Pytorch 60 Minutes Blitz
------------------------

- [Website](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

This is my edited/annotated version of the pytorch 60 minute blitz tutorials. I
heavily comment this series and add examples and descriptions of the functions
used to better grasp what is demonstrated.

This first part focuses on the tensor object and the manipulations that are
possible with the pytorch library.

Tensors are basically like numpy array but you can use them to do
operations on GPUs
'''
from __future__ import print_function
import torch
import numpy as np
import logging
import pprint

def info(MSG, VALUE):
	logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
	pp = pprint.PrettyPrinter(indent=4)
	logging.info(MSG)
	pp.pprint(VALUE)
	print('-'*80)

# Pytorch Matrix Creation

# making empty array
x = torch.empty(5,3)
info('empty array', x)

# matrix filled with random values
x = torch.rand(5,3)
info('random values', x)

# matrix of zeroes of types long
x = torch.zeros(5, 3, dtype=torch.long)
info('matrix of zeros of types long', x)

# making a tensor straigth from data
x = torch.tensor([5.5, 3])
info('creating a tensor with data', x)

# create tensor based on an existing tensor and change the datatype
x = x.new_ones(5,3, dtype=torch.double)

xmod = torch.randn_like(x, dtype=torch.float)
info('create tensor from existing tensor and change the datatype', (x, xmod))

# get size of the matrix
info('getting the tensor size', x.size())

'''
Tensor/Matrix Operations

[Documentation](https://pytorch.org/docs/stable/torch.html)

Three ways of performing additions in pytorch are demonstrated.
'''
# method 1
y = torch.rand(5,3)
info('addition method 1', (x + y))

# method 2
info('addition method 2', (torch.add(x,y)))

# provide output tensor as argument
result = torch.empty(5,3)
torch.add(x, y, out=result)
info('addition method 3', result)

# method 3
# addition can also be done like this, this will change (mutate) matrix "y"
# all torch operations that do this have "_" at the end 
# (i.e. x.copy_(y), x.t_())
y.add_(x)
info('addition method 3', y)


# to print out array index use the numpy notation
'''
Review of numpy array indexing

basic slicing syntax is i:j:k where i=start index j= stop index k=step
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x[1:7:2] = [1, 3, 5]

Negative values for i or j are interpreted as n+i or n+j. Negative k means
we are stepping in decreasing indices.
x[-2: 10] = [8,9]
x[-3: 3: -1] = [7, 6, 5, 4]

If i is not given default is 0 (if k>0) or n-1 (if k<0)
If j is not given default is n (if k>0) or -n-1 (if k<0)
If k is not given default is 1
NOTE: :: is the same as : -> selects all incidices along this axis
x[5:] = [5, 6, 7, 8, 9]

For multidimensional arrays note that x[0,2] = x[0][2]. NOTE: x[0,2] is the
method you should use.
x[start;stop(row),stop(column):step]
	you basically set the axis by with j which is a tuple when working with 2D
	arrays (row, column)
'''
info('indexing tensor like numpy arrays', (x[:,1]))



# to resize array
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8) #-1 -> the size is inferred from the other dimensions (in this case -1 = 2)
info('resizing the tensor with view', (x.size(), y.size(), z.size()))


# 1 element tensor get value with .item
x = torch.randn(1)
info('get the value of single element tensor with .item()', (x.item()))


# Numpy Conversions

a = torch.ones(5)
b = a.numpy()
info('torch tensor to numpy array', (a,b))

# numpy array will also change in value
a.add_(1)
info('operation of the tensor will change the numpy array as well', (a,b))

# going from numpy to torch tensor, operation also affects both matrices
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
info('numpy array to torch tensor, operations also affect both ', (a,b))


# Tensor to GPU
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    info('operation done with cuda', (z))
    info('send to cpu', (z.to("cpu", torch.double)))       # ``.to`` can also change dtype together!

'''
Autograd:  Automatic Differentiation
------------------------------------

The ```autograd``` package is of central importance in pytorch. It provides
automatic differentiation for operations on tensors. Define-by-run framework:
the backprop is defined by how the code is run (every single iteration can be
different).

Define-by-Run
- Network is defined dynamically via the actual forward computation.
- This dynamic definition allows conditionals and loops into the network
  definitions easily. Chainer, PyTorch

How to use it:

torch.Tensor is the central class of the package. Set ```.requires_grad``` ->
```True``` and it will track the operations fone on it. When you finish the
computations, call ```.backward()``` and you will get the gradients computed
automatically. The gradients are stored in ```.grad``` attribute.

To stop tracking histry you call ```.detach()```.

Wrap code with ```with torch.no_grad():``` to disable tracking. This allows to
save memory. This can be useful since maybe you dont't need to use gradients on
some parameters.

The other class which is important is ```Function```: builds a history of the
tensor computations. Each ```Tensor``` has a ```.grad_fn``` which refers to the
```Function``` that created it (NOTE: the tensors created by the user will not
have this ```.grad_fn = None```).
To compute the derivatives you call ```.backward()``` on a ```Tensor``` (if
scalar no arguments, orthewise you specify a ```gradient``` which is a tensor
of matching shape.

[Documentation on functions](https://pytorch.org/docs/stable/autograd.html#function)
'''

# create a tensor that will track computations
x = torch.ones(2,2, requires_grad=True)
info('tensor that tracks computations ', x)
y = x + 2  # perform an operation
info('gradient function after operation', y.grad_fn)
z = y * y * 3 # another operation (do multiplication)
out = z.mean() # and another operation (compute mean)
info('more operations',(z, out))
# if you don't put the flag requires_grad -> False
a = torch.randn(2,2)
a = ((a*3)/(a-1))
info('without the flag, requires_grad defaults to false ', a.requires_grad)
a.requires_grad_(True)
info('flag can be changed after declaration ', a.requires_grad)
b = (a*a).sum()
info('demo of tracking', b.grad_fn)


# gradients:  backpropagation
out.backward()  # NOTE: out contains only a single scalar
                # it is equivalent to -> out.backward(torch.tensor(1.))
info('automatic differentiation of a scalar', x.grad)  #this prints the gradient from d(out)/dx


# A more complex example involving Jacobian matrix and vector product
x = torch.randn(3, requires_grad=True)
y = x*2
while y.data.norm() < 1000:
    y = y*2
info('a vector', y) # so here we have a vector
                # since y is not a scalar we are unable to compute the 
                # full J directly

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
info('automatic differentiation of a vector', x.grad)

# stop the tracking of history
info('gradient tracking', x.requires_grad)
info('gradient tracking', (x ** 2).requires_grad)

with torch.no_grad():
    info('we stop tracking here', (x ** 2).requires_grad)

# use .detach() to create a new tensor with same content but without gradients
info('tensor tracking gradient', x.requires_grad)
y = x.detach()
info('you can create a copy of a tensor that will not track', y.requires_grad)
