import torch.nn as nn
import math
import torchvision
import torch
import numpy as np

__all__ = ['mobilenetv2']


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):

    def __init__(self, channel=320, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(in_planes=channel)
        self.sa = SpatialAttention()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32, ):
        super(CoordAtt, self).__init__()

        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        x_size = torch.tensor(x.shape)
        n, c, h, w = x_size
        inputsz = np.array([h, w])
        outputszh = np.array([h, 1])
        strideszh = np.floor(inputsz / outputszh).astype(np.int32)
        kernelszh = inputsz - (outputszh - 1) * strideszh

        outputszw = np.array([1, w])
        strideszw = np.floor(inputsz / outputszw).astype(np.int32)
        kernelszw = inputsz - (outputszw - 1) * strideszw
        pool_h = nn.AvgPool2d(kernel_size=list(kernelszh), stride=list(strideszh))
        pool_w = nn.AvgPool2d(kernel_size=list(kernelszw), stride=list(strideszw))
        x_h = pool_h(x)
        # x_h = self.pool_h(x)
        # # print(x_h.shape)
        x_w = pool_w(x).permute(0, 1, 3, 2)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        # # # 
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h
        # y = x_h
        return y

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class MobileNetV2(nn.Module):
    __factory = {
        320: torchvision.models.mobilenet_v2,
    }

    def __init__(self, depth=320, num_classes=1000, width_mult=1.0, pretrained=True, matconvnet=None,
                 cut_at_pooling=False):
        super(MobileNetV2, self).__init__()
        self.pretrained = pretrained
        self.matconvnet = matconvnet
        # setting of inverted residual blocks
        mobilenet_v2 = MobileNetV2.__factory[depth](pretrained=pretrained)
        self.feature_dim = 320
        self.cut_at_pooling = cut_at_pooling
        layers = list(mobilenet_v2.features.children())[:-1]
        # layers.append(CoordAtt(320,320))
        self.features = nn.Sequential(*layers)  # capture only feature part and remove last relu and maxpool
        self.gap = nn.AdaptiveMaxPool2d(1)
        # building last several layers
        # output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        # self.conv = conv_1x1_bn(input_channel, output_channel)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(output_channel, num_classes)

        self._init_params()

    def forward(self, x):
        x = self.features(x)
        if self.cut_at_pooling:
            return x
        pool_x = self.gap(x)
        pool_x = pool_x.view(pool_x.size(0), -1)

        return pool_x, x

    # x = self.conv(x)
    #  x = self.avgpool(x)
    # x = x.view(x.size(0), -1)
    # x = self.classifier(x)
    def _init_params(self):
        # optional load pretrained weights from matconvnet
        if self.matconvnet is not None:
            self.features.load_state_dict(
                torch.load(models_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')))
            self.pretrained = True

        # for m in self.modules():
        #   if isinstance(m, nn.Conv2d):
        #       n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #       m.weight.data.normal_(0, math.sqrt(2. / n))
        #       if m.bias is not None:
        #           m.bias.data.zero_()
        #   elif isinstance(m, nn.BatchNorm2d):
        #       m.weight.data.fill_(1)
        #       m.bias.data.zero_()
        #   elif isinstance(m, nn.Linear):
        #       m.weight.data.normal_(0, 0.01)
        #       m.bias.data.zero_()


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(depth=320, **kwargs)