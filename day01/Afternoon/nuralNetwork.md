Neural Networks in PyTorch (Example):

Let's build a simple neural network using PyTorch to classify handwritten digits (MNIST dataset).

Import necessary modules

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

Define the neural network model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # input layer (784) -> hidden layer (128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # hidden layer (128) -> output layer (10)

    def forward(self, x):
        x = self.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x

Load the MNIST dataset

transform = transforms.ToTensor()
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

Initialize the model, loss function, and optimizer

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

Train the model

for epoch in range(10):  # loop over the dataset multiple times
    for x, y in trainloader:
        # forward pass
        x = x.view(-1, 784)  # flatten the input image
        y_pred = net(x)
        loss = criterion(y_pred, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

This example demonstrates a basic neural network in PyTorch, including:

- Define a neural network model using nn.Module
- Load a dataset (MNIST) using datasets and DataLoader
- Initialize the model, loss function, and optimizer
- Train the model using a for loop

This is just a starting point, and you can build upon this example to create more complex neural networks and experiments in PyTorch!