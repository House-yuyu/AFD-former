import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

def conv(in_channels, out_channels,kernel_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

class VRCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.con1 = nn.Sequential(conv(in_channels, 64, 5),
                    nn.ReLU(inplace=True))

        self.con2 = nn.Sequential(conv(64, 16, 5),
                    nn.ReLU(inplace=True))

        self.con3 = nn.Sequential(conv(64, 32, 3),
                    nn.ReLU(inplace=True))

        self.con4 = nn.Sequential(conv(48, 16, 3),
                    nn.ReLU(inplace=True))

        self.con5 = nn.Sequential(conv(48, 32, 1),
                    nn.ReLU(inplace=True))

        self.con6 = conv(48, out_channels, 3)

    def forward(self, x):
        x1 = self.con1(x)
        x2 = self.con2(x1)
        x3 = self.con3(x1)

        conca1 = torch.cat((x2, x3), 1)
        x4 = self.con4(conca1)
        x5 = self.con5(conca1)

        conca2 = torch.cat((x4, x5), 1)
        x6 = self.con6(conca2)

        out = x + x6

        return out

if __name__ == "__main__":
    model = VRCNN()
    # print(model)

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)  # 输入是元组，且不需要batch
    print('flops: ', flops, 'params: ', params)
