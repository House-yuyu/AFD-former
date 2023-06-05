import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class conv5x5(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
        m = [
            nn.Conv2d(
                in_channels, in_channels, kernel_size, stride=stride,
                padding=(kernel_size // 2), bias=bias),
            nn.PReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=(kernel_size // 2), bias=bias),

        ]

        super(conv5x5, self).__init__(*m)


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False, act=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv3x3(in_channels, out_channels))
            if bn:
                m.append(nn.BatchNorm2d(64))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class MSRAB_module(nn.Module):
    def __init__(self):
        super(MSRAB_module, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(2, 1), nn.BatchNorm2d(1))
        self.conv2 = nn.Sequential(conv5x5(2, 1), nn.BatchNorm2d(1))
        self.conv3 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(1))
        self.conv4 = conv1x1(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out1 = self.conv1(out)
        out2 = self.conv2(out)
        out3 = self.conv3(out)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv4(out)
        out = self.sigmoid(out)
        out = torch.mul(residual, out)
        return out


class MSRAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSRAB, self).__init__()
        self.conv3x3_1 = nn.Sequential(conv3x3(in_channels, out_channels), nn.BatchNorm2d(out_channels))  # 和c通道保持一致，在空间维度上bn
        self.conv3x3_2 = nn.Sequential(conv3x3(out_channels * 2, out_channels * 2), nn.BatchNorm2d(out_channels * 2))
        self.conv5x5_1 = nn.Sequential(conv5x5(in_channels, out_channels), nn.BatchNorm2d(out_channels))
        self.conv5x5_2 = nn.Sequential(conv5x5(out_channels * 2, out_channels * 2))
        self.conv1x1 = conv1x1(out_channels * 4, out_channels)
        self.msrab_layer = MSRAB_module()
        self.prelu = nn.PReLU()

    def forward(self, x):
        out1 = self.conv3x3_1(x)
        out2 = self.conv5x5_1(x)
        out = torch.cat([out1, out2], dim=1)
        out3 = self.conv3x3_2(out)
        out4 = self.conv5x5_2(out)
        out = torch.cat([out3, out4], dim=1)
        out = self.conv1x1(out)
        out = self.msrab_layer(out)
        out += x
        return out


class GlobalStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalStream, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.conv1 = conv3x3(in_channels * 4, out_channels)
        self.RB1 = ResidualBlock(out_channels, out_channels)
        self.RB2 = ResidualBlock(out_channels, out_channels)
        self.RB3 = ResidualBlock(out_channels, out_channels)
        self.RB4 = ResidualBlock(out_channels, out_channels)
        self.RB5 = ResidualBlock(out_channels, out_channels)
        self.RB6 = ResidualBlock(out_channels, out_channels)
        self.RB7 = ResidualBlock(out_channels, out_channels)
        self.RB8 = ResidualBlock(out_channels, out_channels)
        self.RB9 = ResidualBlock(out_channels, out_channels)
        self.RB10 = ResidualBlock(out_channels, out_channels)
        self.RB11 = ResidualBlock(out_channels, out_channels)
        self.RB12 = ResidualBlock(out_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels * 4)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        out = self.pixel_unshuffle(x)
        out = self.conv1(out)
        out = self.RB1(out)
        out = self.RB2(out)
        out = self.RB3(out)
        out = self.RB4(out)
        out = self.RB5(out)
        out = self.RB6(out)
        out = self.RB7(out)
        out = self.RB8(out)
        out = self.RB9(out)
        out = self.RB10(out)
        out = self.RB11(out)
        out = self.RB12(out)
        out = self.conv2(out)
        out = self.pixel_shuffle(out)
        return out


class LocalStream(nn.Module):  # 3->64
    def __init__(self, in_channels, out_channels):
        super(LocalStream, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.msrab1 = MSRAB(64, 64)
        self.msrab2 = MSRAB(64, 64)
        self.msrab3 = MSRAB(64, 64)
        self.msrab4 = MSRAB(64, 64)
        self.msrab5 = MSRAB(64, 64)
        self.msrab6 = MSRAB(64, 64)

    def forward(self, x):
        out = self.conv1(x)
        residual = out

        out = self.msrab1(out)
        out = self.msrab2(out)
        out = self.msrab3(out)
        out = self.msrab4(out)
        out = self.msrab5(out)
        out = self.msrab6(out)
        out = torch.cat([residual, out], dim=1)
        return out


class TSAN(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(TSAN, self).__init__()
        self.global_stream = GlobalStream(inchannels, 64)
        self.local_stream = LocalStream(inchannels, 64)
        self.conv1 = conv1x1(192, 64)
        self.conv2 = conv3x3(64, outchannels)

    def forward(self, x):
        out1 = self.local_stream(x)
        out2 = self.global_stream(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


if __name__ == '__main__':

    model = TSAN(3,3)
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)



