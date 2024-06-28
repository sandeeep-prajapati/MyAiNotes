Building a simple neural network involves several steps:

1. Import necessary libraries: Import libraries like PyTorch, NumPy, and Matplotlib.
2. Define the model: Define a class that inherits from PyTorch's nn.Module. Define the layers and activation functions in the __init__ method.
3. Define the forward pass: Define the forward pass method that takes input data and passes it through the layers.
4. Define the loss function and optimizer: Define the loss function and optimizer to use for training.
5. Train the model: Train the model using the training data.
6. Test the model: Test the model using the testing data.

Here's a simple example of a neural network that classifies handwritten digits (MNIST dataset):


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the model
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

# Load the MNIST dataset
transform = transforms.ToTensor()
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

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

# Test the model
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for x, y in testloader:
        x = x.view(-1, 784)  # flatten the input image
        y_pred = model(x)
        test_loss += criterion(y_pred, y).item()
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y).sum().item()

accuracy = correct / len(testloader.dataset)
print(f'Test loss: {test_loss / len(testloader)}')
print(f'Test accuracy: {accuracy:.2f}%')


This is a basic example of building a neural network in PyTorch. You can modify the model, loss function, and optimizer to suit your specific needs.