#https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/#setting-hyperparameters

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset paths
dataset_paths = {
    'ecoset': {
        'train': '/home/wallacelab/trained_modules/ecoset/downloads/5197c8300ef32f04a398233d0ef225a4e271beb85737e46dc1d6e9fc7eb33e2b/train',
        'val': '/home/wallacelab/trained_modules/ecoset/downloads/5197c8300ef32f04a398233d0ef225a4e271beb85737e46dc1d6e9fc7eb33e2b/val',
        'test': '/home/wallacelab/trained_modules/ecoset/downloads/5197c8300ef32f04a398233d0ef225a4e271beb85737e46dc1d6e9fc7eb33e2b/test',
    },
    'imagenet': {
        'train': '/home/wallacelab/trained_modules/ILSVRC2012_img_train.tar',
        'val': '/home/wallacelab/trained_modules/ILSVRC2012_img_val.tar',
    }
}

def load_dataset(data_dir, batch_size, is_train=True, is_valid=False, is_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return data_loader

class ResidualBlock(nn.Module):
    expansion = 1  # Expansion factor for ResNet
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * ResidualBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * ResidualBlock.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Hyperparameters
hyperparams = {
    'batch_size': 64,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.001,
}

# Model, Loss, and Optimizer
model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=1000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=hyperparams['learning_rate'], momentum=hyperparams['momentum'], weight_decay=hyperparams['weight_decay'])

# Dataset Selection and DataLoader Initialization
dataset_name = 'imagenet'  # or 'ecoset'
train_loader = load_dataset(dataset_paths[dataset_name]['train'], hyperparams['batch_size'], is_train=True)
valid_loader = load_dataset(dataset_paths[dataset_name]['val'], hyperparams['batch_size'], is_valid=True)

# Ecoset 'test' directory is for testing and not present in ImageNet
if dataset_name == 'ecoset':
    test_loader = load_dataset(dataset_paths[dataset_name]['test'], hyperparams['batch_size'], is_test=True)

# Train and Validate the Model
for epoch in range(hyperparams['num_epochs']):
    # Training Loop
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass, backward pass, and optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{hyperparams["num_epochs"]}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Test the Model
if dataset_name == 'ecoset':
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
