import torch
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out

# layers [2, 2, 2, 2]
# Bottleneck
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_dims = 3
        self.inplanes = 64
        # stride: 2
        self.conv1 = nn.Conv2d(self.in_dims, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # stride: 4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stride: 4
        self.layer1 = self.make_layer(block, 64, layers[0])
        # stride: 8
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        # stride: 16
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        # stride: 32
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        # init model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, inplanes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != inplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, inplanes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(inplanes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, inplanes, stride, downsample))
        self.inplanes = inplanes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, inplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feats = x['images']
        fs = self.conv1(feats)
        fs = self.bn1(fs)
        fs = self.relu(fs)
        fs = self.maxpool(fs)

        l1 = self.layer1(fs)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        x.update({'feats': [l2, l3, l4]})
        return x

def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model