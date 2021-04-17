import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, dim=32, pooling=True):
        super().__init__()
        self.inplanes = dim
        self.pooling = pooling
        self.conv1 = nn.Conv2d(1, dim, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, dim, layers[0])
        self.layer2 = self._make_layer(BasicBlock, dim * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, dim * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, dim * 8, layers[3], stride=2)

        #         self.pool = AdaptiveConcatPool2d()
        #         self.pool = torch.nn.AdaptiveMaxPool2d(1)
        if pooling:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.size()[1] > 1:
            x = x[:, 0:1, :, :]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.pooling:
            x = self.pool(x)

        return x


class LinearClassifier(nn.Module):
    def __init__(self, dim, num_cls):
        super().__init__()
        self.fc_clf = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features=dim, out_features=num_cls, bias=True),)

    def forward(self, x):
        clf = self.fc_clf(x)
        return clf


class LeadWishCNN(nn.Module):
    def __init__(self, num_cls, layers=[2, 2, 2, 2], dim=64):
        super().__init__()

        self.num_cls = num_cls
        self.feature = ResNet(BasicBlock, layers, dim=dim)  # 34:[3,4,6,3]
        self.classifier = LinearClassifier(dim * 8 * 12, num_cls)

    def forward(self, x):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.view(N * C, 1, H, W)
        x = self.feature(x)
        x = x.reshape(N, -1)
        out = self.classifier(x)

        return out


class EmbedCNN(nn.Module):
    def __init__(self, out_dim=128, layers=[2, 2, 2, 2], dim=64, pooling=False):
        super().__init__()

        self.feature = ResNet(BasicBlock, layers, dim=dim, pooling=pooling)  # 34:[3,4,6,3]
        #  self.project = nn.Linear(dim*8, out_dim)

    def forward(self, x):
        B = x.size(0)
        out = self.feature(x)
        #  x = x.reshape(B, -1)
        #  out = self.project(x)
        return out


if __name__ == "__main__":
    net = EmbedCNN(out_dim=1, dim=32)
    import ipdb

    ipdb.set_trace()
    x = torch.randn(5, 1, 64, 64)
    x = torch.autograd.Variable(x)
    out = net(x)
    print(out.shape)
