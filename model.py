import torch as t
from torch import nn
from blocks import WPM_Input, FLM, MUM
import math
from unet_double_plus import UnetDoublePlus
from torch.nn import functional as F


class MetaSR(nn.Module):

    def __init__(self, k, inC, outC, FLM_use_unet, LR_size, use_interpolate_branch):
        super(MetaSR, self).__init__()
        self.use_interpolate_branch = use_interpolate_branch
        self.FLM_use_unet = FLM_use_unet
        self.WPM_input = WPM_Input()
        if FLM_use_unet:
            self.FLM = UnetDoublePlus(out_channels=inC)
            assert (LR_size[0] / 2 == LR_size[0] // 2) and (LR_size[0] / 4 == LR_size[0] // 4) and (LR_size[0] / 8 == LR_size[0] // 8) and (LR_size[0] / 16 == LR_size[0] // 16), "if FLM_use_unet is True, LR_size[0] // (2 ** i) should equal to LR_size[0] / (2 ** i), i=1~4"
            assert (LR_size[1] / 2 == LR_size[1] // 2) and (LR_size[0] / 4 == LR_size[1] // 4) and (LR_size[1] / 8 == LR_size[1] // 8) and (LR_size[1] / 16 == LR_size[1] // 16), "if FLM_use_unet is True, LR_size[1] // (2 ** i) should equal to LR_size[1] / (2 ** i), i=1~4"
        else:
            self.FLM = FLM(out_channels=inC)
        self.mum = MUM(k=k, inC=inC, outC=outC)
        if use_interpolate_branch:
            self.to_3_channels = nn.Sequential(
                nn.Conv2d(in_channels=outC, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.1)
            )
            self.last_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            self.to_3_channels = nn.Sequential(
                nn.Conv2d(in_channels=outC, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )

    def forward(self, I_LR, r):
        HR_size = (math.floor(I_LR.size()[-2] * r), math.floor(I_LR.size()[-1] * r))
        offset_vector = self.WPM_input(r, HR_size).cuda(self.to_3_channels.__getitem__(0).bias.device.index)  # (HW, 3)
        F_LR = self.FLM(I_LR)  # (N, inC, I_LR_H, I_LR_W)
        meta_upscale_result = self.mum(F_LR, offset_vector, r, HR_size)
        result = self.to_3_channels(meta_upscale_result)
        if self.use_interpolate_branch:
            interpolate_result = F.interpolate(I_LR, HR_size, mode="bicubic", align_corners=False)
            result = interpolate_result + result
            result = self.last_conv(result)
        return result


if __name__ == "__main__":
    LR_size = (64, 64)
    r = 1.2
    model = MetaSR(k=5, inC=16, outC=32, FLM_use_unet=False, use_interpolate_branch=True, LR_size=(25, 25)).cuda(0)
    d = t.randn(2, 3, LR_size[0], LR_size[1]).cuda(0)
    result = model(d, r)
    print(result.size())