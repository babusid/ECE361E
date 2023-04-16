import torch
import torch.nn as nn
from torch.nn import ReLU6, ReLU
import torch.functional as F



class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_constant=6):
        super(Bottleneck, self).__init__()
        self.stride = stride
        #Expansion stage, 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, expansion_constant*in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expansion_constant*in_channels)
        self.relu1 = ReLU6()
    
        #Depthwise convolution, 3x3 convolution
        self.conv2 = nn.Conv2d(expansion_constant*in_channels, expansion_constant*in_channels, kernel_size=3, stride=stride, padding=1, groups=expansion_constant*in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion_constant*in_channels)
        self.relu2 = ReLU6()
    
        #Pointwise convolution, 1x1 convolution, apparently no final activation according to paper's architecture
        self.conv3 = nn.Conv2d(expansion_constant*in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        #Skip connection pointwise convolution
        if (self.stride == 1):
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
    def forward(self, x):
        #1. Expand the input channels to expansion times the input channels
        #2. Batchnorm
        #3. ReLU6
        #4. Perform a depthwise convolution with stride = 1 and padding = 1, kernel size = 3
        #5. Batchnorm
        #6. ReLU6
        #7. Perform a pointwise convolution with stride = 1 and padding = 0, kernel size = 1
        #8. ReLU6
        #9. Skip Connection (1x1 convolution with stride = stride and padding = 0, kernel size = 1)
        
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if(self.stride == 1):
            x += self.conv4(residual)
        return x



class MobileNetv2(nn.Module):
    #(out_channels, stride, expansion)
    #Transcribed from Table 2 of Mobilenetv2 paper
    cfg = [
        (16,1,1),
        (24,2,6),
        (24,1,6),
        (32,2,6),
        # (32,1,6), # Comment out for: -6
        # (32,1,6),
        # (64,2,6),
        # (64,1,6), # Comment out for: -6
        # (64,1,6), # Comment out for: -6
        # (64,1,6),
        # (96,1,6),
        # (96,1,6), # Comment out for: -6
        # (96,1,6), # Comment out for: -6
        # (160,2,6),
        # (160,1,6), # Comment out for: -6
        # (160,1,6),
        # (320,1,6),
    ]
    def __init__(self, num_classes=10):
        super(MobileNetv2, self).__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = ReLU()
        self.layers = self._make_layers(in_planes=32)
        # self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool2d((2,2))
        # self.conv3 = nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(32*4*4, num_classes)
    
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x[0]
            stride = x[1]
            expansion = x[2]
            layers.append(Bottleneck(in_planes, out_planes, stride, expansion))
            in_planes = out_planes
        
        # return layers
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.relu1(out)      
        
        out = self.layers(out)
        
        # out = self.conv2(out)
        
        out = self.avgpool(out)
        
        # out = self.conv3(out) 
        
        out = out.view(out.size(0), -1)    

        out = self.linear(out)
        
        return out
    
if __name__ == '__main__':
    m = MobileNetv2()
    print(m)
    from torchinfo import summary
    summary(m, input_size=(128, 3, 32, 32))


    dummy = torch.rand((128, 3, 32, 32))
    out = m(dummy)
    # print(out)