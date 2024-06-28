To fine-tune a pre-trained model on a new dataset (e.g., CIFAR-10) in PyTorch, follow these steps:

1. Load the pre-trained model:

model = torchvision.models.resnet50(pretrained=True)

1. Load the new dataset (e.g., CIFAR-10):

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

1. Freeze some layers (optional):

for param in model.parameters():
    param.requires_grad = False

This will freeze all layers, but you can selectively unfreeze specific layers by setting requires_grad to True.

1. Add a new classifier (if needed):

num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

This adds a new fully connected layer (classifier) on top of the pre-trained model.

1. Define a loss function and optimizer:

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

1. Fine-tune the model:

for epoch in range(5):  # Adjust the number of epochs
    for batch in trainloader:
        images, labels = batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

This will fine-tune the pre-trained model on the new dataset (CIFAR-10).

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

This will evaluate the fine-tuned model on the test set and print the test loss and accuracy.