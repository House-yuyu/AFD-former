import torch
import torch.nn as nn
import torch.nn.init as init
import math
from ptflops import get_model_complexity_info
import torch.nn.functional as F
'''
Residual Dense Network for Image Super-Resolution
https://arxiv.org/abs/1802.08797
'''


class RDN(nn.Module):
    def __init__(self, channel=3, rdb_conv_num=8):
        super(RDN, self).__init__()
        self.SFF1 = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.SFF2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.RDB1 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB2 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB3 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB4 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB5 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB6 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB7 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB8 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)

        self.RDB9 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB10 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB11 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB12 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB13 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB14 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB15 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)
        self.RDB16 = RDB(nb_layers=rdb_conv_num, input_dim=64, growth_rate=64)

        self.GFF1 = nn.Conv2d(in_channels=64 * 16, out_channels=64, kernel_size=1, padding=0)  # 1*1融合concat
        self.GFF2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # ->F_gf

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=channel, kernel_size=3, padding=1)

    def forward(self, x):
        f_ = self.SFF1(x)
        f_0 = self.SFF2(f_)

        f_1 = self.RDB1(f_0)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        f_4 = self.RDB4(f_3)
        f_5 = self.RDB5(f_4)
        f_6 = self.RDB6(f_5)
        f_7 = self.RDB7(f_6)
        f_8 = self.RDB8(f_7)

        f_9 = self.RDB8(f_8)
        f_10 = self.RDB8(f_9)
        f_11 = self.RDB8(f_10)
        f_12 = self.RDB8(f_11)
        f_13 = self.RDB8(f_12)
        f_14 = self.RDB8(f_13)
        f_15 = self.RDB8(f_14)
        f_16 = self.RDB8(f_15)

        f_D = torch.cat((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8,
                         f_9, f_10, f_11, f_12, f_13, f_14, f_15, f_16), 1)

        f_1x1 = self.GFF1(f_D)
        f_GF = self.GFF2(f_1x1)

        f_DF = f_GF + f_

        f_convRes = self.conv2(f_DF)
        f_res = f_convRes + x

        return f_res


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, nb_layers, input_dim, growth_rate):
        super(RDB, self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers, input_dim, growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels=input_dim + nb_layers * growth_rate,
                                 out_channels=growth_rate,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

    def _make_layer(self, nb_layers, input_dim, growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(input_dim + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv1x1(out)
        return out + x


if __name__ == '__main__':
    model = RDN(3, 8)
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
