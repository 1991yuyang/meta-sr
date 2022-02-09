from torch import nn
import torch as t
from torchvision import models


class CostumLoss(nn.Module):

    def __init__(self):
        super(CostumLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        loss_net = models.vgg16(pretrained=True)
        self.feat1 = nn.Sequential(*list(loss_net.features.children())[:4])
        self.feat2 = nn.Sequential(*list(loss_net.features.children())[4:9])
        self.feat3 = nn.Sequential(*list(loss_net.features.children())[9:16])
        self.feat4 = nn.Sequential(*list(loss_net.features.children())[16:23])
        self.feat1.eval()
        self.feat2.eval()
        self.feat3.eval()
        self.feat4.eval()
        self.set_requires_grad(self.feat1)
        self.set_requires_grad(self.feat2)
        self.set_requires_grad(self.feat3)
        self.set_requires_grad(self.feat4)

    def forward(self, sr, I_hr):
        loss1 = self.L1(sr, I_hr)

        sr_feat1 = self.feat1(sr)
        sr_feat2 = self.feat2(sr_feat1)
        sr_feat3 = self.feat3(sr_feat2)
        sr_feat4 = self.feat4(sr_feat3)

        I_hr_feat1 = self.feat1(I_hr)
        I_hr_feat2 = self.feat2(I_hr_feat1)
        I_hr_feat3 = self.feat3(I_hr_feat2)
        I_hr_feat4 = self.feat4(I_hr_feat3)

        feat1_loss = self.mse(sr_feat1, I_hr_feat1)
        feat2_loss = self.mse(sr_feat2, I_hr_feat2)
        feat3_loss = self.mse(sr_feat3, I_hr_feat3)
        feat4_loss = self.mse(sr_feat4, I_hr_feat4)
        loss2 = feat1_loss + feat2_loss + feat3_loss + feat4_loss

        total_loss = loss1 + loss2
        return total_loss

    def set_requires_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    # criterion = CostumLoss()
    vgg = models.vgg16(pretrained=True)
    print(list(vgg.children()))
    # vgg.eval()
    # for param in vgg.parameters():
    #     print(param.requires_grad)