PyTorch Implementation:

PyTorch is a popular deep learning framework that provides a dynamic computation graph and automatic differentiation. Here's an example of how to implement a simple neural network in PyTorch:


import torch
import torch.nn as nn
import torch.optim as optim

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

# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
for epoch in range(10):  # loop over the dataset multiple times
    for x, y in trainloader:
        # forward pass
        x = x.view(-1, 784)  # flatten the input image
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


This implementation defines a simple neural network with one hidden layer, uses the ReLU activation function, and trains the model using the SGD optimizer and Cross-Entropy loss function.

Note:

- nn.Module is the base class for all neural network modules in PyTorch.
- nn.Linear is a module for a linear layer (fully connected layer).
- nn.ReLU is a module for the ReLU activation function.
- nn.CrossEntropyLoss is a module for the Cross-Entropy loss function.
- optim.SGD is a module for the SGD optimizer.
- model.parameters() returns a list of all model parameters.
- loss.backward() computes the gradients of the loss with respect to the model parameters.
- optimizer.step() updates the model parameters based on the gradients and the learning rate.