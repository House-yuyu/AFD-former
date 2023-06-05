import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)


class LMSA(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1_1 = conv(in_channels, 64, 1)
        self.stride_con = nn.Conv2d(in_channels, 64, kernel_size=2, stride=2, padding=0)
        self.conv3_1 = conv(1, 1, 3)
        self.conv3_2 = conv(1, 1, 3)
        self.conv3_3 = conv(1, 1, 3)

        self.conv1_2 = conv(3, 64, 1)
        self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x1 = self.conv1_1(x)
        stride = self.stride_con(x1)
        maxout, _ = torch.max(stride, dim=1, keepdim=True)
        conv3_1 = self.conv3_1(maxout)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_2(conv3_2)

        cat = torch.cat([conv3_1, conv3_2, conv3_3], dim=1)
        conv1_2 = self.conv1_2(cat)  #
        up = self.bilinear(conv1_2)
        sigmod = self.sigmoid(up)
        out = torch.mul(residual, sigmod)
        return out


class FDAB(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1_1 = nn.Sequential(conv(in_channels, 64, 1),
                                     nn.PReLU())
        self.conv3_1 = nn.Sequential(conv(in_channels, 64, 3),
                                     nn.PReLU())

        self.conv1_2 = nn.Sequential(conv(64, 64, 1),
                                     nn.PReLU())
        self.conv3_2 = nn.Sequential(conv(64, 64, 3),
                                     nn.PReLU())

        self.conv1_3 = nn.Sequential(conv(64, 64, 1),
                                     nn.PReLU())
        self.conv3_3 = nn.Sequential(conv(64, 64, 3),
                                     nn.PReLU())

        self.conv1_4 = nn.Sequential(conv(64, 64, 1),
                                     nn.PReLU())

        self.conv1_5 = nn.Sequential(conv(64*4, 64, 1),
                                     nn.PReLU())

        self.LMSA = LMSA()

    def forward(self, x):
        residual = x

        conv1_1 = self.conv1_1(x)
        conv3_1 = self.conv3_1(x)
        input2 = x + conv3_1

        conv2_1 = self.conv1_2(input2)
        conv2_3 = self.conv3_2(input2)
        input3 = input2 + conv2_3

        conv3_1 = self.conv1_3(input3)
        conv3_3 = self.conv3_3(input3)
        input4 = input3 + conv3_3

        conv4_1 = self.conv1_4(input4)

        concat = torch.cat([conv1_1, conv2_1, conv3_1, conv4_1], dim=1)

        conv5_1 = self.conv1_5(concat)
        LMSA = self.LMSA(conv5_1)

        out = LMSA + residual

        return out


class RSB(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.RSB = nn.Sequential(conv(in_channels, 64, 3), nn.PReLU(), conv(64, 64, 3))

    def forward(self, x):
        residual = x

        out = residual + self.RSB(x)

        return out


class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle, self).__init__()
        self.num_groups = num_groups
    def forward(self, x: torch.FloatTensor):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
         # (batch_size, num_groups, chs_per_group, h, w)
        x = x.transpose(1, 2)  # dim_1 and dim_2
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out


class RFFB(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.RSB = RSB()
        self.channel_shuffle1 = Channel_Shuffle(128)  # nn.ChannelShuffle could not run with cuda
        self.channel_shuffle2 = Channel_Shuffle(128)

        self.conv1_1 = conv(128, 64, 1)
        self.conv1_2 = conv(128, 64, 1)

        self.weight1 = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))
        self.weight2 = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))

        self.weight1.data.fill_(1)
        self.weight2.data.fill_(1)

    def forward(self, x):
        residual = x

        RSB1 = self.RSB(x)
        RSB2 = self.RSB(RSB1)
        RSB3 = self.RSB(RSB2)

        cat1 = torch.cat([RSB1, RSB2], dim=1)
        channel_shuffle1 = self.channel_shuffle1(cat1)
        conv1 = self.conv1_1(channel_shuffle1)

        cat2 = torch.cat([conv1, RSB3], dim=1)
        channel_shuffle2 = self.channel_shuffle2(cat2)
        conv2 = self.conv1_2(channel_shuffle2)

        out = conv2*self.weight1 + residual*self.weight2

        return out


class RDEN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.local_conv3 = conv(in_channels, 64, 3)

        self.FDAB1 = FDAB()
        self.FDAB2 = FDAB()
        self.FDAB3 = FDAB()
        self.FDAB4 = FDAB()
        self.FDAB5 = FDAB()
        self.FDAB6 = FDAB()

        self.unshuffle_conv3 = nn.Sequential(nn.PixelUnshuffle(2), conv(in_channels*4, 64, 3))

        self.RFFB1 = RFFB()
        self.RFFB2 = RFFB()
        self.RFFB3 = RFFB()

        self.shuffle_conv3 = nn.Sequential(conv(64, 256, 3), nn.PixelShuffle(2))

        self.conv1 = conv(64*3, 64, 1)
        self.recon_conv3 = conv(64, out_channels, 3)

    def forward(self, x):

        local_conv3 = self.local_conv3(x)

        FDAB1 = self.FDAB1(local_conv3)
        FDAB2 = self.FDAB2(FDAB1)
        FDAB3 = self.FDAB3(FDAB2)
        FDAB4 = self.FDAB4(FDAB3)
        FDAB5 = self.FDAB5(FDAB4)
        FDAB6 = self.FDAB6(FDAB5)

        unshuffle = self.unshuffle_conv3(x)

        RFFB1 = self.RFFB1(unshuffle)
        RFFB2 = self.RFFB2(RFFB1)
        RFFB3 = self.RFFB3(RFFB2)

        shuffle = self.shuffle_conv3(RFFB3)

        cat = torch.cat([local_conv3, FDAB6, shuffle], dim=1)

        conv1 = self.conv1(cat)

        out = self.recon_conv3(conv1)

        return out


if __name__ == "__main__":

    model = RDEN()
    # print(model)

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)  # 输入是元组，且不需要batch
    print('flops: ', flops, 'params: ', params)