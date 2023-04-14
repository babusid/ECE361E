import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out

class MobileNetv1(nn.Module):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetv1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        # quantization
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(1, -1)
        x = self.linear(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self, qat=False):
        fuse_func = torch.ao.quantization.fuse_modules_qat if qat else (
                    torch.ao.quantization.fuse_modules)
        fuse_func(self, 
                ['conv1', 'bn1', 'relu1'], 
                inplace=True)
        for m in self.modules():
            if type(m) == Block:
                fuse_func(m, 
                        [['conv1', 'bn1', 'relu1'], 
                        ['conv2', 'bn2', 'relu2']],
                        inplace=True)

if __name__ == '__main__':
    m = MobileNetv1()
    m.eval()
    m.fuse_model(True)
    print(m)
