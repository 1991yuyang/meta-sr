import torch
from torch import nn
from torch.nn import functional as F


class make_dense(nn.Module):

    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, out_channels, nDenselayer=3, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_out = nn.Conv2d(in_channels=nChannels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        out = self.conv_out(out)
        return out


if __name__ == "__main__":
    model = RDB(3)
    d = torch.randn(2, 3, 256, 256)
    output = model(d)
    print(output.size())