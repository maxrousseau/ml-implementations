#!/usr/bin/env python3
'''
Deep Learning with Python - Francois Chollet
===

Chapter 1 - What is Deep Learning
---

The goal of training neural networks is to find the correct weights for each
layer. The loss function gives us a score based on the prediction (Y') and the ground
truth (Y). The optimizer will then use this score to adjust the weights to
attempt to improve the performance of the network. This is done through
backpropagation (gradient descent).

The author presents a history of machine learning. Main takeaways are that
currently two techniques should be mastered: gradient boosting (xdgboost) for
shallow learning (also works well with structured data) and deep learning for
perceptual problems (complex data structures such as images). Familiarization
XGboost and Keras libraries will be important.

Discussion about the current state of deep learning and its surge in efficacy in
the past few years (compute and data).

Chapter 2 - The mathematical building blocks of neural networks
---

Going through the MNIST problem as an example. Ten handwritten digits (0-9)
stored in 28x28 images to be classified.

Tensor: It is an array of data. Tensors can be scalars (0 dimension), vectors
(1 dimension), or matrices (2 dimensions) and higher dimensional (3D tensors or other).
x0 = np.array(12)
x1 = np.array([12, 4, 5, 14, 9]) #a unidimensional tensor of rank 5
x2 = np.array([[1, 3, 4],
               [4, 5, 8]])

Attributes of tensors:
- number of axes (rank) matrix has 2, 3D tensor has 3
- shape of the tensor is a tuple of integers. (3, 5) or (3, 3, 5) or (5,) vector
and () scalar.
- data type (float32, float64, uint8, char #rare)

For MNIST our training data is (60000, 28, 28) -> a 3nD tensor, 60 000 matrices
of 28x28 uint8 (0-255 grayscale)

Tensor operations:
        element wise operations
        broadcasting
        dot product
        reshaping

Stochastic Gradient Descent:
        gradient is the derivatives of a tensor operation

Backpropagation:

Training step:
        epochs
        mini-batch


In keras, to build and run a network you must define the following:
- layers (flow of input to output data)
- loss function (how the network measures how well it has performed)
- optimizer (how the network will update itself based on how well it did ^)
- metrics (logging the progress of the network)

Chapter 3 - Getting Started with Neural Networks
---



'''

from tensorflow.keras.datasets import mnist

from tensorflow.keras import models
from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(len(test_labels))


# we have to change the input shape to fit of first layer 512
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255


# changing the labels to categoricals, for classification purposes
# this will be explained in chapter 3
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 28*28 = 512, we build a fully connected NN here (no convolutiions yet)
# declaring the architecture by adding the layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# the compile step where we declare the optimizera and loss function
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# fitting the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# test and print accuracy
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
