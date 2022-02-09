import torch as t
from torch import nn
from torch.nn import functional as F
import math
from rdn import RDB
"""
LR_size is invariable in training and test process and specified in config file
high resolution image size equal to floor(LR_size * r)
"""


class WPM_Input(nn.Module):

    """
    get input vector of weight prediction module
    """
    def __init__(self):
        """

        :param HR_size: hight resolution image size like (H, W)
        """
        super(WPM_Input, self).__init__()

    def forward(self, r, HR_size):
        """

        :param r: scale factor
        :return:
        """
        row_index = t.floor(t.arange(HR_size[0] * HR_size[1]).type(t.FloatTensor) / HR_size[1]).type(t.FloatTensor)
        colum_index = t.transpose(t.floor(t.arange(HR_size[0] * HR_size[1]).type(t.FloatTensor) / HR_size[0]).type(t.FloatTensor).view(HR_size[::-1]), dim0=1, dim1=0).flatten()
        row_value = (row_index / r - t.floor(row_index / r)).view((-1, 1))
        colum_value = (colum_index / r - t.floor(colum_index / r)).view((-1, 1))
        scale_value = (t.ones(HR_size[0] * HR_size[1]).type(t.FloatTensor) / r).view((-1, 1))
        offset_vector = t.cat([row_value, colum_value, scale_value], dim=1).type(t.FloatTensor)
        return offset_vector


# class WPM(nn.Module):
#
#     """
#     weight prediction module
#     """
#     def __init__(self, k, inC, outC):
#         """
#
#         :param k: int, for example k represent k * k size kernel
#         :param inC: channels of lower resolution image features
#         :param outC: count of kernel
#         """
#         super(WPM, self).__init__()
#         self.k = k
#         self.inC = inC
#         self.outC = outC
#         self.linear = nn.Sequential(
#             nn.Linear(in_features=3, out_features=256, bias=True),
#             nn.ReLU(),
#             nn.Linear(in_features=256, out_features=k ** 2 * inC * outC, bias=True)
#         )
#
#     def forward(self, offset_vector, HR_size):
#         return self.linear(offset_vector).view((HR_size[0], HR_size[1], self.outC, self.inC, self.k, self.k))  # (H, W, outC, inC, k, k)


class WPM(nn.Module):

    """
    weight prediction module
    """
    def __init__(self, k, inC, outC):
        """

        :param k: int, for example k represent k * k size kernel
        :param inC: channels of lower resolution image features
        :param outC: count of kernel
        """
        super(WPM, self).__init__()
        self.k = k
        self.inC = inC
        self.outC = outC
        self.linear = nn.Sequential(
            nn.Linear(in_features=3, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=k ** 2 * inC * outC, bias=True)
        )

    def forward(self, offset_vector, HR_size):
        return self.linear(offset_vector).view((HR_size[0] * HR_size[1], self.outC, self.inC, self.k, self.k))  # (H * W, outC, inC, k, k)


class FLM(nn.Module):

    """
    feature learning module
    """
    def __init__(self, out_channels):
        super(FLM, self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.block1 = RDB(nChannels=out_channels, out_channels=128)
        self.block2 = RDB(nChannels=128, out_channels=256)
        self.block3 = RDB(nChannels=256, out_channels=512)
        self.block4 = RDB(nChannels=512, out_channels=1024)
        self.last_conv = nn.Conv2d(in_channels=1024 + 512 + 256 + 128, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, I_LR):
        """

        :param I_LR: low resolution image tensor
        :return:
        """
        head_conv_result = self.head_conv(I_LR)
        block1_output = self.block1(head_conv_result)
        block2_output = self.block2(block1_output)
        block3_output = self.block3(block2_output)
        block4_output = self.block4(block3_output)
        cat_result = t.cat([block1_output, block2_output, block3_output, block4_output], dim=1)
        last_conv_result = self.last_conv(cat_result)
        output = head_conv_result + last_conv_result
        return output


# class MUM(nn.Module):
#
#     """
#     meta upscale module
#     """
#     def __init__(self, k, inC, outC):
#         super(MUM, self).__init__()
#         self.k = k
#         self.inC = inC
#         self.outC = outC
#         self.weight_predict_model  = WPM(k=k, inC=inC, outC=outC)
#
#     def forward(self, F_LR, offset_vector, r, HR_size):
#         F_LR_PAD = F.pad(F_LR, (self.k - 1, self.k - 1, self.k - 1, self.k - 1), "constant", value=0)
#         predicted_weight = self.weight_predict_model(offset_vector, HR_size)    # (H, W, outC, inC, k, k)
#         result = []
#         for i in range(predicted_weight.size()[0]):
#             raw_result = []
#             for j in range(predicted_weight.size()[1]):
#                 i_ = i // r
#                 j_ = j // r
#                 pixel_weight = predicted_weight[i, j]  # (outC, inC, k, k)
#                 f_lr_part = F_LR_PAD[:, :, int(i_ + 1 - self.k // 2):int(i_ + 1 - self.k // 2 + self.k), int(j_ + 1 - self.k // 2):int(j_ + 1 - self.k // 2 + self.k)]  # (N, inC, self.k, self.k)
#                 pixel_of_sr = F.conv2d(f_lr_part, pixel_weight)  # (N, outC, 1, 1)
#                 raw_result.append(pixel_of_sr)
#             result.append(t.cat(raw_result, dim=3))
#         result = t.cat(result, dim=2)
#         return result


class MUM(nn.Module):

    """
    meta upscale module
    """
    def __init__(self, k, inC, outC):
        super(MUM, self).__init__()
        self.k = k
        self.inC = inC
        self.outC = outC
        self.weight_predict_model  = WPM(k=k, inC=inC, outC=outC)

    def forward(self, F_LR, offset_vector, r, HR_size):
        F_LR_PAD = F.pad(F_LR, ((self.k - 1) // 2, (self.k - 1) // 2, (self.k - 1) // 2, (self.k - 1) // 2), "constant", value=0)
        predicted_weight = self.weight_predict_model(offset_vector, HR_size)    # (H * W, outC, inC, k, k)
        part_results = []
        for i in range(HR_size[0]):
            for j in range(HR_size[1]):
                i_ = i // r
                j_ = j // r
                f_lr_part = F_LR_PAD[:, :, int(i_):int(self.k + i_), int(j_):int(j_ + self.k)]  # (N, inC, self.k, self.k)
                part_results.append(f_lr_part.unsqueeze(0))
        part_results = t.cat(part_results, dim=0)
        part_results = part_results.unsqueeze(2)
        predicted_weight = predicted_weight.unsqueeze(1)
        result = t.sum(part_results * predicted_weight, dim=[3, 4, 5]).permute(dims=[1, 2, 0]).view((-1, self.outC, HR_size[0], HR_size[1]))
        return result


if __name__ == "__main__":
    model = WPM_Input()
    offset_vector = model(2, (5, 10))
    print(offset_vector)
    print(offset_vector.size())