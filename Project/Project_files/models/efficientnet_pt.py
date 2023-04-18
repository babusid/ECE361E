import torch.nn as nn

class MBConvBlock(nn.module):
    def __init__(self, in_planes, out_planes, stride=1, expand_ratio=1):
        pass
    def forward(self, x):
        pass    


class EfficientNetB0(nn.Module):
    '''Implements EfficientNetB0 as seen in Table 1 here: https://arxiv.org/pdf/1905.11946.pdf'''

    def __init__(self, num_classes=10):
        pass
    def _make_layers(self, in_planes):
        pass
    def forward(self, x):
        pass