Here's an example of using a pre-trained ResNet50 model in PyTorch:

import torch
import torchvision
import torchvision.transforms as transforms

# Load the pre-trained ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load a sample image
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
image = transform(Image.open('image.jpg'))

# Add a batch dimension (batch size = 1)
image = image.unsqueeze(0)

# Forward pass to get the predictions
outputs = model(image)

# Get the predicted class index
_, predicted = torch.max(outputs, 1)

# Print the predicted class index
print(predicted.item())

This code:

1. Loads the pre-trained ResNet50 model using torchvision.models.resnet50(pretrained=True).
2. Sets the model to evaluation mode using model.eval().
3. Loads a sample image using Image.open and applies transformations (resizing, center cropping, and tensor conversion) using transforms.Compose.
4. Adds a batch dimension (batch size = 1) to the image tensor using unsqueeze(0).
5. Performs a forward pass to get the predictions using model(image).
6. Gets the predicted class index using torch.max(outputs, 1).
7. Prints the predicted class index using print(predicted.item()).

Note: This is just a simple example, and you may need to adjust the code depending on your specific use case.