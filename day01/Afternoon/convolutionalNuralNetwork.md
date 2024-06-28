Convolutional Neural Networks (CNNs):

A Convolutional Neural Network (CNN) is a type of neural network architecture that uses convolutional and pooling layers to extract features from images and other spatial data.

Key aspects:

- Convolutional layers: Apply filters to small regions of the input data to detect local patterns
- Pooling layers: Downsample the data to reduce spatial dimensions and the number of parameters
- Flatten layer: Flatten the feature maps into a 1D vector for classification or regression

Example:

Suppose we want to build a CNN to classify images into two classes: dogs and cats.

- Input: 3-channel color images of size 256x256
- Convolutional layer 1:
    - Filter size: 3x3
    - Number of filters: 16
    - Activation function: ReLU
- Max pooling layer 1:
    - Pool size: 2x2
    - Stride: 2
- Convolutional layer 2:
    - Filter size: 3x3
    - Number of filters: 32
    - Activation function: ReLU
- Max pooling layer 2:
    - Pool size: 2x2
    - Stride: 2
- Flatten layer
- Dense layer (classification):
    - Number of units: 2 (one for each class)
    - Activation function: Softmax

PyTorch example:

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32*64*64, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 32*64*64)
        x = self.fc(x)
        return x

# Initialize the CNN model
model = CNN()

This is a basic example of a CNN architecture for image classification. There are many variations and techniques to improve the performance of CNNs, such as batch normalization, dropout, and transfer learning.