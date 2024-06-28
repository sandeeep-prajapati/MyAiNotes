Here's an example of using transfer learning for image classification in PyTorch:

1. Load a pre-trained model (e.g., ResNet50):

model = torchvision.models.resnet50(pretrained=True)

1. Freeze the model's weights:

for param in model.parameters():
    param.requires_grad = False

1. Add a new classifier layer:

num_classes = 10  # Replace with your number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

1. Load your dataset and data loader:

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.ImageFolder(root='path/to/train/directory', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='path/to/test/directory', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

1. Define a loss function and optimizer:

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

1. Train the model:

for epoch in range(5):  # Adjust the number of epochs
    for batch in trainloader:
        images, labels = batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

1. Evaluate the model on the test set:

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for batch in testloader:
        images, labels = batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
accuracy = correct / len(testloader.dataset)
print(f'Test loss: {test_loss / len(testloader)}')
print(f'Test accuracy: {accuracy:.2f}%')

This is a basic example of using transfer learning for image classification in PyTorch. You can adjust the model, dataset, and hyperparameters to suit your specific needs.