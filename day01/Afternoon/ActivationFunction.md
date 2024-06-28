Activation functions are used in neural networks to introduce non-linearity into the output of a neuron or layer. This is necessary because most real-world relationships between inputs and outputs are non-linear.

Common activation functions:

1. ReLU (Rectified Linear Unit): f(x) = max(0, x)
    - Fast and computationally efficient
    - Non-linearity is introduced by setting negative values to 0
2. Sigmoid: f(x) = 1 / (1 + exp(-x))
    - Maps the input to a value between 0 and 1
    - Used for binary classification problems
3. Tanh (Hyperbolic Tangent): f(x) = 2 / (1 + exp(-2x)) - 1
    - Maps the input to a value between -1 and 1
    - Similar to sigmoid, but with a larger range
4. Softmax: f(x) = exp(x) / sum(exp(x))
    - Used for multi-class classification problems
    - Maps the input to a probability distribution over all classes
5. Leaky ReLU: f(x) = max(alpha * x, x)
    - A variation of ReLU with a small slope for negative values
6. Swish: f(x) = x * sigmoid(x)
    - A recently introduced activation function that can be seen as a self-gated version of ReLU

The choice of activation function depends on the specific problem and the design of the neural network. Some common guidelines are:

- Use ReLU or Leaky ReLU for hidden layers
- Use Sigmoid or Softmax for output layers (depending on the problem)
- Use Tanh for models that require a larger range of values

In PyTorch, you can use these activation functions as follows:

import torch
import torch.nn as nn

# ReLU
relu = nn.ReLU()

# Sigmoid
sigmoid = nn.Sigmoid()

# Tanh
tanh = nn.Tanh()

# Softmax
softmax = nn.Softmax(dim=1)

# Leaky ReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# Swish
swish = nn.Swish()
