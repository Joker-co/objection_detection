import torch
import torch.nn as nn

# layers [2, 2, 2, 2]
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
        self.layer1 = self.make_layer(block, self.inplanes, layers[0])



    def forward(self, x):
        feats = x['images']
        fs = self.conv1(feats)
        fs = self.bn1(fs)
        fs = self.relu(fs)
        fs = self.maxpool(fs)

        l1 = self.layer1(fs)