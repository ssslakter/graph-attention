import torch
import torch.nn as nn
import timm
import os


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1, self.bn1 = conv3x3(inplanes, planes, stride), nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2, self.bn2 = conv3x3(planes, planes), nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)




class AttnResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_chans=3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build the 4 stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)

    @classmethod
    def load_from_timm(cls, model_name, num_classes=1000, pretrained=False):
        # Configuration mapping for common models
        configs = {
            "resnet18": {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            "resnet34": {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            "resnet50": {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            "resnet101": {"block": Bottleneck, "layers": [3, 4, 23, 3]},
        }

        if model_name not in configs:
            raise ValueError(f"Config for {model_name} not defined. Add it to the configs dict.")

        # 1. Create our custom model
        config = configs[model_name]
        model = cls(block=config["block"], layers=config["layers"], num_classes=num_classes)

        # 2. Grab weights from timm
        print(f"Fetching pretrained weights for {model_name}...")
        timm_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        # 3. Load state dict
        model.load_state_dict(timm_model.state_dict())
        return model

