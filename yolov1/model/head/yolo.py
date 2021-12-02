import torch
import torch.nn as nn

class Yolov1Head(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(Yolov1Head, self).__init__()
        self.num_classes = num_classes
        # objectness, classes, locations
        self.out_channels = 1 + self.num_classes + 4
        self.pred = nn.Conv2d(inplanes, self.out_channels, kernel_size=1)

    def forward(self, x):
        feat = x['feats']

        out = self.pred(feat)
        x.update({'feats': out})
        return x