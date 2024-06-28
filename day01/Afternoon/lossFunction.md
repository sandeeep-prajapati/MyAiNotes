Loss functions are used to measure the difference between the model's predictions and the true labels. The goal of training a model is to minimize the loss function.

Common loss functions:

1. Cross-Entropy: Measures the difference between predicted probabilities and true labels.
    - Example: loss = nn.CrossEntropyLoss()(predictions, labels)
2. Mean Squared Error (MSE): Measures the average squared difference between predicted values and true labels.
    - Example: loss = nn.MSELoss()(predictions, labels)
3. Binary Cross-Entropy: Measures the difference between predicted probabilities and true labels for binary classification.
    - Example: loss = nn.BCELoss()(predictions, labels)
4. Mean Absolute Error (MAE): Measures the average absolute difference between predicted values and true labels.
    - Example: loss = nn.L1Loss()(predictions, labels)
5. Huber Loss: A combination of MSE and MAE, used for regression tasks.
    - Example: loss = nn.HuberLoss()(predictions, labels)

PyTorch examples:

import torch
import torch.nn as nn

# Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()
predictions = torch.tensor([[0.5, 0.5], [0.8, 0.2]])
labels = torch.tensor([1, 0])
loss = criterion(predictions, labels)

# Mean Squared Error (MSE) Loss
criterion = nn.MSELoss()
predictions = torch.tensor([[0.5, 0.5], [0.8, 0.2]])
labels = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
loss = criterion(predictions, labels)

# Binary Cross-Entropy Loss
criterion = nn.BCELoss()
predictions = torch.tensor([[0.5], [0.8]])
labels = torch.tensor([[1], [0]])
loss = criterion(predictions, labels)

These loss functions are commonly used in various machine learning tasks, such as classification, regression, and segmentation. The choice of loss function depends on the specific problem and the type of task.