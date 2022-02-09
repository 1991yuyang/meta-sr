from torch.utils import data
from torchvision import transforms as T
import math
from PIL import Image
import os
import numpy as np
from torchvision.transforms import functional as F
from numpy import random as rd


class MySet(data.Dataset):

    def __init__(self, LR_size, hr_img_dir):
        self.LR_size = LR_size
        self.hr_img_dir = hr_img_dir
        self.transformer_get_lr = T.Compose([
            T.Resize(self.LR_size),
            T.ToTensor()
        ])
        self.img_pths = [os.path.join(self.hr_img_dir, name) for name in os.listdir(hr_img_dir)]

    def __getitem__(self, index):
        img_pth = self.img_pths[index]
        img = Image.open(img_pth)
        img = self.data_aug(img)
        lr = self.transformer_get_lr(img)
        return lr, img

    def __len__(self):
        return len(self.img_pths)

    def data_aug(self, pil_img):
        #  1.rotate
        if rd.uniform(0, 1) < 0.5:
            pil_img = F.rotate(pil_img, rd.randint(-30, 30))
        #  2.H flip
        if rd.uniform(0, 1) < 0.5:
            pil_img = F.hflip(pil_img)
        #  3.V flip
        if rd.uniform(0, 1) < 0.5:
            pil_img = F.vflip(pil_img)
        return pil_img

def get_collate(r_range, r_step):
    def my_collate(batch):
        r = round(float(rd.choice(np.arange(r_range[0] + r_step, r_range[1] + r_step, r_step))), len(str(r_step).split(".")[-1]))
        data = [item[0].unsqueeze(0) for item in batch]
        lr_size = data[0].size()[-2:]
        hr_size = (int(math.floor(lr_size[0] * r)), int(math.floor(lr_size[1] * r)))
        target = [F.to_tensor(F.resize(item[1], hr_size, Image.BILINEAR)).unsqueeze(0) for item in batch]
        return [data, target, r]
    return my_collate


def make_loader(LR_size, hr_img_dir, batch_size, collate, num_workers):
    loader = iter(data.DataLoader(MySet(LR_size, hr_img_dir), batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    import torch as t
    collate_fn = get_collate([1, 4], 0.01)
    loader = make_loader((200, 200), "/home/yuyang/data/bluetooth/train_image", 4, collate_fn, 0)
    to_pil = T.ToPILImage()
    for d, l, r in loader:
        print(t.max(t.cat(l, dim=0)))
        print(t.min(t.cat(l, dim=0)))
        # to_pil(l[0].squeeze(0)).show()
        # print("---------------------")
        # input()