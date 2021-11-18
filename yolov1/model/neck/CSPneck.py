import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1,
                 stride=1, padding=0, dilation=1, groups=1, leaky=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.act = nn.LeakyReLU(0.1, inplace=True) if leaky else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out

class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x3 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat([x, x1, x2, x3], dim=1)
        return out

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, shortcut=True, groups=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_planes = int(planes * expansion)
        self.conv1 = Conv(inplanes, hidden_planes, kernel_size=1)
        self.conv2 = Conv(hidden_planes, planes, kernel_size=3, padding=1, groups=groups)
        self.add = shortcut and inplanes == planes
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add:
            out = out + x
        return out

class BottleneckCSP(nn.Module):
    def __init__(self, inplanes, planes, nblocks=1, shortcut=True, groups=1, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        hidden_planes = int(planes * expansion)
        self.conv1 = Conv(inplanes, hidden_planes, kernel_size=1)
        self.blocks = nn.Sequential(*[
            Bottleneck(hidden_planes, hidden_planes, shortcut, groups, expansion=1) for _ in range(nblocks)
        ])
        self.conv3 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inplanes, hidden_planes, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(2 * hidden_planes)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv4 = Conv(2 * hidden_planes, planes, kernel_size=1)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.blocks(out1)
        out1 = self.conv3(out1)
        out2 = self.conv2(x)

        out = torch.cat((out1, out2), dim=1)
        out = self.bn(out)
        out = self.act(out)
        out = self.conv4(out)
        return out

class SAM(nn.Module):
    def __init__(self, inplanes):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.act(attention)
        return x * attention

class CSPNeck(nn.Module):
    def __init__(self, inplanes, planes):
        self.SPP = nn.Sequential(
            Conv(inplanes, planes, kernel_size=1),
            SPP(),
            BottleneckCSP(planes * 4, planes * 2, nblocks=1, shortcut=False)
        )
        self.SAM = SAM(planes * 2)
        self.ConvCSP = BottleneckCSP(planes * 2, planes * 2, nblocks=3, shortcut=False)

    def forward(self, x):
        _, _, feat = x['feats']

        out = self.SPP(feat)
        out = self.SAM(out)
        out = self.ConvCSP(out)
        x.update({'feats': out})
        return x