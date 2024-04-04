#https://blog.paperspace.com/alexnet-pytorch/#alexnet

#imagenet and econet dataset have been splitted into training and validation subset

#i havent extracted the imagenet dataset yet

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths for the datasets
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

def get_loader(data_dir, batch_size, is_train=True, shuffle=True):
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def initialize_alexnet(num_classes=1000):  # Default to ImageNet classes
    model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
    hyperparams = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.005,
        'num_epochs': 20,
    }
    return model, criterion, optimizer, hyperparams

# AlexNet Model
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Select dataset
dataset_name = 'ecoset'  # Change to 'imagenet' for ImageNet dataset
num_classes = 567 if dataset_name == 'ecoset' else 1000
model, criterion, optimizer, hyperparams = initialize_alexnet(num_classes=num_classes)

# Loaders
train_loader = get_loader(dataset_paths[dataset_name]['train'], hyperparams['batch_size'], is_train=True)
val_loader = get_loader(dataset_paths[dataset_name]['val'], hyperparams['batch_size'], is_train=False)
if dataset_name == 'ecoset':
    test_loader = get_loader(dataset_paths[dataset_name]['test'], hyperparams['batch_size'], is_train=False)

# Train the model
total_step = len(train_loader)
for epoch in range(hyperparams['num_epochs']):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{hyperparams["num_epochs"]}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
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

    print(f'Accuracy of the model on the test images: {100 * correct / total} %')




# /Users/owusunp/miniconda3/envs/gpt3/bin/python /Users/owusunp/Desktop/pygui/alexnet.py
# (gpt3) (base) owusunp@Princes-MacBook-Pro pygui % /Users/owusunp/miniconda3/envs/gpt3/bin/python /Users/owusu
# np/Desktop/pygui/alexnet.py
# Files already downloaded and verified
# Files already downloaded and verified
# Files already downloaded and verified
# Epoch [1/20], Step [100/704], Loss: 2.3021
# Epoch [1/20], Step [200/704], Loss: 2.3024
# Epoch [1/20], Step [300/704], Loss: 2.3020
# Epoch [1/20], Step [400/704], Loss: 2.2985
# Epoch [1/20], Step [500/704], Loss: 2.3007
# Epoch [1/20], Step [600/704], Loss: 2.3061
# Epoch [1/20], Step [700/704], Loss: 2.2842
# Epoch [2/20], Step [100/704], Loss: 2.2414
# Epoch [2/20], Step [200/704], Loss: 2.1955
# Epoch [2/20], Step [300/704], Loss: 2.1511
# Epoch [2/20], Step [400/704], Loss: 2.1003
# Epoch [2/20], Step [500/704], Loss: 2.0036
# Epoch [2/20], Step [600/704], Loss: 1.8060
# Epoch [2/20], Step [700/704], Loss: 1.8877
# Epoch [3/20], Step [100/704], Loss: 1.6772
# Epoch [3/20], Step [200/704], Loss: 1.8230
# Epoch [3/20], Step [300/704], Loss: 1.5710
# Epoch [3/20], Step [400/704], Loss: 1.7011
# Epoch [3/20], Step [500/704], Loss: 1.7111
# Epoch [3/20], Step [600/704], Loss: 1.5864
# Epoch [3/20], Step [700/704], Loss: 1.6974
# Epoch [4/20], Step [100/704], Loss: 1.4226
# Epoch [4/20], Step [200/704], Loss: 1.6453
# Epoch [4/20], Step [300/704], Loss: 1.7277
# Epoch [4/20], Step [400/704], Loss: 1.5413
# Epoch [4/20], Step [500/704], Loss: 1.4092
# Epoch [4/20], Step [600/704], Loss: 1.2706
# Epoch [4/20], Step [700/704], Loss: 1.4923
# Epoch [5/20], Step [100/704], Loss: 1.4390
# Epoch [5/20], Step [200/704], Loss: 1.4169
# Epoch [5/20], Step [300/704], Loss: 1.1842
# Epoch [5/20], Step [400/704], Loss: 1.5150
# Epoch [5/20], Step [500/704], Loss: 1.3993
# Epoch [5/20], Step [600/704], Loss: 1.3146
# Epoch [5/20], Step [700/704], Loss: 1.4044
# Epoch [6/20], Step [100/704], Loss: 1.5611
# Epoch [6/20], Step [200/704], Loss: 1.4236
# Epoch [6/20], Step [300/704], Loss: 1.2788
# Epoch [6/20], Step [400/704], Loss: 1.3056
# Epoch [6/20], Step [500/704], Loss: 1.3357
# Epoch [6/20], Step [600/704], Loss: 1.3802
# Epoch [6/20], Step [700/704], Loss: 1.3264


# Epoch [7/20], Step [100/704], Loss: 1.1784
