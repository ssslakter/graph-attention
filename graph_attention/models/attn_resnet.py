import torch
import torch.nn as nn, torch.nn.functional as F
import timm
from einops import rearrange, einsum, reduce
import os


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
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


class ChannelAttentionResBlock(BasicBlock):
    """Channel-wise attention for 2D feature maps."""

    def __init__(self, inplanes, planes, stride=1, downsample=None, region_size=4):
        super().__init__(inplanes, planes, stride, downsample)
        self.gamma = 10
        self.k = region_size
        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Linear(self.k**2, self.k**2 * 2, bias=False)
        self.attn_scale = nn.Parameter(torch.tensor(1e-5))

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        b, c, h, w = x.shape

        z = F.adaptive_avg_pool2d(x, (self.k, self.k))
        z = rearrange(z, "b c k1 k2 -> b c (k1 k2)")

        qk = self.to_qk(z).chunk(2, dim=-1)
        q, k = qk

        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) / self.k)

        x_flat = rearrange(x, "b c h w -> b c (h w)")
        out = torch.matmul(attn, x_flat)

        out = rearrange(out, "b c (h w) -> b c h w", h=h, w=w)
        out = (1 - self.attn_scale) * x + (self.attn_scale * out)

        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
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
    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        attn_layer_indices: list[int] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        region_size: int = 4,
    ):
        super().__init__()
        self.inplanes = 64
        self.region_size = region_size
        self.attn_layer_indices = attn_layer_indices if attn_layer_indices is not None else []
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build the 4 stages
        self.layer1 = self._make_layer(block, 64, layers[0], block_idx=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, block_idx=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, block_idx=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, block_idx=3)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: type[BasicBlock], planes, blocks, stride=1, block_idx=0):
        downsample = None
        if block_idx in self.attn_layer_indices:
            current_block = ChannelAttentionResBlock
            block_kwargs = {"region_size": self.region_size}
        else:
            current_block = block
            block_kwargs = {}
        if stride != 1 or self.inplanes != planes * current_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * current_block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * current_block.expansion),
            )
        layers = [current_block(self.inplanes, planes, stride, downsample=downsample, **block_kwargs)]
        self.inplanes = planes * current_block.expansion
        for _ in range(1, blocks):
            layers.append(current_block(self.inplanes, planes, **block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)

    @classmethod
    def load_from_timm(cls, model_name, num_classes=1000, pretrained=True, attn_layer_indices=None, **kwargs):
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
        model = cls(
            block=config["block"],
            layers=config["layers"],
            attn_layer_indices=attn_layer_indices,
            num_classes=num_classes,
            **kwargs,
        )

        # 2. Grab weights from timm
        print(f"Fetching pretrained weights for {model_name}...")
        timm_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        # 3. Load state dict
        model.load_state_dict(timm_model.state_dict(), strict=False)
        return model
