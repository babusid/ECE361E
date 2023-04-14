import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub
import math

class DepthWiseConv(nn.Module):
    def __init__(self, in_planes, stride):
        super(DepthWiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes))
    def forward(self, x):
        return self.layers(x)

class PointWiseConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(PointWiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())
    def forward(self, x):
        return self.layers(x)
    
class Shuffle(nn.Module):
    '''Shuffle channels'''
    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.groups, c // self.groups, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(n, -1, h, w)
        return x
    
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, downsample):
        super(InvertedResidual, self).__init__()
        self.downsample = downsample
        self.stride = 2 if downsample else 1

        out_planes_downsample = out_planes // 2
        b2_in_planes_downsampled = in_planes if downsample else out_planes_downsample

        if downsample:
            self.branch1 = nn.Sequential(
                DepthWiseConv(in_planes, self.stride),
                PointWiseConv(in_planes, out_planes_downsample))
        self.branch2 = nn.Sequential(
            PointWiseConv(b2_in_planes_downsampled, out_planes_downsample),
            DepthWiseConv(out_planes_downsample, self.stride),
            PointWiseConv(out_planes_downsample, out_planes_downsample))
        self.shuffle = Shuffle(3)
    def forward(self, x):
        if self.downsample:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            x1, x2 = self._channel_split(x)
            out = torch.cat((x1, self.branch2(x2)), 1)
        return self.shuffle(out)
    @staticmethod
    def _channel_split(x):
        return x[:, :(x.size(1) // 2), :, :], x[:, (x.size(1) // 2):, :, :]




class ShuffleNetV2(nn.Module):
    stage_cfg = [3, 7, 3]   # repeats per stage
    output_channels = [24, 48, 96, 192, 1024]   # 0.5x multiplier

    def __init__(self, input_size=32, num_classes=10):
        super(ShuffleNetV2, self).__init__()

        self.stage1 = nn.Sequential(
                nn.Conv2d(3, self.output_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.output_channels[0]),
                nn.ReLU())

        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)

        self.stage5 = PointWiseConv(self.output_channels[3], self.output_channels[4])
        self.gpool = nn.AvgPool2d(input_size // (2**3))
        self.linear = nn.Linear(self.output_channels[-1], num_classes)

        # quantization
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    # stage_num = 2, 3, 4
    def _make_stage(self, stage_num):
        stage_idx, output_idx = stage_num - 2, stage_num - 1
        return nn.Sequential(
            *([InvertedResidual(self.output_channels[output_idx - 1], self.output_channels[output_idx], True)] + 
              [InvertedResidual(self.output_channels[output_idx], self.output_channels[output_idx], False) 
                    for _ in range(self.stage_cfg[stage_idx])]))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.gpool(x)
        x = x.view(-1, self.output_channels[-1])
        x = self.linear(x)
        return x
    
    def fuse_model(self, qat=False):
        fuse_func = torch.ao.quantization.fuse_modules_qat if qat else (
                    torch.ao.quantization.fuse_modules)
        

if __name__ == '__main__':
    m = ShuffleNetV2()
    print(m)

    dummy = torch.rand((128, 3, 32, 32))
    out = m(dummy)
    print(out)
