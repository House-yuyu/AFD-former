import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

def conv(in_channels, out_channels,kernel_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

class QECNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.con1 = nn.Sequential(conv(in_channels, 128, 9),
                    nn.PReLU())

        self.con2 = nn.Sequential(conv(128, 64, 7),
                    nn.PReLU())

        self.con3 = nn.Sequential(conv(64, 64, 3),
                    nn.PReLU())

        self.con4 = nn.Sequential(conv(64, 32, 1),
                    nn.PReLU())

        self.con5 = conv(32, out_channels, 5)  # orignal method train gray image, output channel is 1

    def forward(self, x):
        x1 = self.con1(x)
        x2 = self.con2(x1)
        x3 = self.con3(x2)
        x4 = self.con4(x3)
        out = self.con5(x4)

        return out

if __name__ == "__main__":
    model = QECNN()
    # print(model)

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)  # 输入是元组，且不需要batch
    print('flops: ', flops, 'params: ', params)
