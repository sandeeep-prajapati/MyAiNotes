nn.Module in PyTorch:

nn.Module is the base class for all neural network modules in PyTorch. It provides a framework for defining and composing neural network layers, including convolutional layers, recurrent layers, and linear layers.

Key Features:

- Module: A module is a building block of a neural network, representing a single layer or a group of layers.
- Parameters: Modules can have parameters, which are tensors that are optimized during training.
- Forward: Modules define a forward method, which computes the output of the module given an input.
- Training: Modules can be in training mode or evaluation mode, which affects the behavior of some modules (e.g., dropout, batch normalization).

Common Methods:

- __init__: Initializes the module and its parameters.
- forward: Computes the output of the module.
- parameters(): Returns a list of the module's parameters.
- train(): Sets the module to training mode.
- eval(): Sets the module to evaluation mode.

Examples of nn.Module subclasses:

- nn.Linear: A linear (fully connected) layer.
- nn.Conv2d: A 2D convolutional layer.
- nn.ReLU: A rectified linear unit (ReLU) activation function.
- nn.BatchNorm2d: A batch normalization layer.

By inheriting from nn.Module, you can define custom neural network layers and modules, making it easy to build and compose complex neural networks in PyTorch.

Example:

import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(5, 3)

    def forward(self, x):
        return self.linear(x)

module = MyModule()
input = torch.randn(1, 5)
output = module(input)

This example defines a custom module MyModule containing a linear layer, and demonstrates how to instantiate and use it.