To load pre-trained models in PyTorch, you can use the torchvision.models module. Here are some examples:

1. ResNet50:

model = torchvision.models.resnet50(pretrained=True)

1. VGG16:

model = torchvision.models.vgg16(pretrained=True)

1. DenseNet121:

model = torchvision.models.densenet121(pretrained=True)

1. InceptionV3:

model = torchvision.models.inception_v3(pretrained=True)

1. MobileNetV2:

model = torchvision.models.mobilenet_v2(pretrained=True)

1. Bert (language model):

model = torchvision.models.bert.BertModel.from_pretrained('bert-base-uncased')

These models are downloaded from the PyTorch model zoo and are ready to use for inference or fine-tuning.

You can also specify the model architecture and weights separately:

model = torchvision.models.resnet50()
model.load_state_dict(torchvision.models.resnet50(pretrained=True).state_dict())
``
This way, you can use a custom model architecture and still use the pre-trained weights.

Remember to call `model.eval()` to set the model to evaluation mode before using it for inference.
