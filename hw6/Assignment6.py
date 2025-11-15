import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np


def compute_num_parameters(net:nn.Module):
    """compute the number of trainable parameters in *net* e.g., ResNet-34.  
    Return the estimated number of parameters Q1. 
    """
    num_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return num_para


def CIFAR10_dataset_a(mode):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    if mode == "train":
        return trainloader
    else:
        return testloader


class GAPNet(nn.Module):
    """
    Insert your code here
    """
    def __init__(self):
        super(GAPNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(6, 10, kernel_size=(5, 5), stride=(1, 1))
        self.gap = nn.AvgPool2d(kernel_size=10, stride=10, padding=0)
        self.fc = nn.Linear(in_features=10, out_features=10, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 6, 14, 14]
        x = F.relu(self.conv2(x))             # [B, 10, 10, 10]
        x = self.gap(x)                       # [B, 10, 1, 1]
        # flatten to feed into FC layer
        x = x.view(x.size(0), -1)             # [B, 10]
        x = self.fc(x)                        # [B, 10]
        return x
    
def train_GAPNet():
    """
    Insert your code here
    """
    trainloader = CIFAR10_dataset_a(mode='train')
    net = GAPNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/10], Step [{i}/{len(trainloader)}], Loss: {loss.item():.4f}")

    print("Finished Training")
    torch.save(net.state_dict(), "./Gap_net_10epoch.pth")


def eval_GAPNet():
    """
    Insert your code here
    """
    testloader = CIFAR10_dataset_a(mode='test')
    net = GAPNet()
    net.load_state_dict(torch.load("./Gap_net_10epoch.pth"))
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")

def backbone():
    """
    Insert your code here, Q3
    """
    resnet18 = models.resnet18(pretrained=True)
    # remove the final fully connected layer
    modules = list(resnet18.children())[:-1]
    resnet18_feat = nn.Sequential(*modules)
    resnet18_feat.eval()
    # load and preprocess the image
    img = Image.open("cat_eye.jpg")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    # get features
    with torch.no_grad():
        features = resnet18_feat(batch_t) 
    return features

def transfer_learning():
    """
    Insert your code here, Q4
    """
    trainloader = CIFAR10_dataset_a(mode='train')
    testloader = CIFAR10_dataset_a(mode='test')
    resnet18 = models.resnet18(pretrained=True)
    # modify the last layer to output 10 classes
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)
    # freeze all the weights except the last layer
    for param in resnet18.parameters():
        param.requires_grad = False
    for param in resnet18.fc.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/10], Step [{i}/{len(trainloader)}], Loss: {loss.item():.4f}")

    print("Finished Training")
    torch.save(resnet18.state_dict(), "./Res_net_10epoch.pth")

    # evaluate the model
    resnet18.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = resnet18(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")

'''
DenseNet.  Write PyTorch code to implement MobileNet.  Derive your class 
from nn.Module, and implement both __init__() and forward() methods.  Be sure to 
include batch norm and ReLU.  Run your network on an image to be sure that all the 
dimensions are correct; you do not have to check that the output makes sense.  (It will 
not, since the network is not trained.) Q5
'''

class MobileNetV1(nn.Module):
    """Define MobileNetV1 please keep the strucutre of the class Q5"""
    '''
    Args:
        ch_in (int): number of input channels (e.g., 3 for RGB).
        n_classes (int): number of output classes (e.g., 1000 for ImageNet, 10 for CIFAR-10).
    '''
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            # 3x3 conv -> BN -> ReLU
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=3, stride=stride,
                          padding=1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            # Depthwise 3x3 conv -> BN -> ReLU
            # + Pointwise 1x1 conv -> BN -> ReLU
            return nn.Sequential(
                # depthwise
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride,
                          padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # pointwise
                nn.Conv2d(inp, oup, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        # macth the standard MobileNetV1 config.
        self.features = nn.Sequential(
            conv_bn(ch_in, 32, stride=2),      # 224x224 -> 112x112
            conv_dw(32, 64, stride=1),
            conv_dw(64, 128, stride=2),        # 112x112 -> 56x56
            conv_dw(128, 128, stride=1),
            conv_dw(128, 256, stride=2),       # 56x56 -> 28x28
            conv_dw(256, 256, stride=1),
            conv_dw(256, 512, stride=2),       # 28x28 -> 14x14
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 1024, stride=2),      # 14x14 -> 7x7
            conv_dw(1024, 1024, stride=1),
        )

        # global average pooling to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # final linear
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)          # [B, 1024, 1, 1]
        x = x.view(x.size(0), -1)    # [B, 1024]
        x = self.fc(x)               # [B, n_classes]
        return x
        

    
if __name__ == '__main__':
    #Q1
    resnet34 = models.resnet34(pretrained=True)
    num_para = compute_num_parameters(resnet34)
    #print(num_para)
    #Q2
    #train_GAPNet()
    #eval_GAPNet()
    # Q3: backbone()
    features = backbone()
    #print(features.shape)
    # Q5
    ch_in = 3
    n_classes = 1000
    model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)