import torch.nn as nn

cfg = {
    'VGG5': [16, 32, 64, 128, 256],
}


class VGG5(nn.Module):
    def __init__(self, vgg_name="VGG5"):
        super(VGG5, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(cfg[vgg_name][-1], 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        stride = 2
        for x in cfg:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=stride),
                       nn.BatchNorm2d(x),
                        nn.ReLU()]
            in_channels = x
        return nn.Sequential(*layers)
    
