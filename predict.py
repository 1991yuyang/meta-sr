import torch as t
from torchvision import transforms as T
import json
from torch import nn
import os
import math
from PIL import Image
from model import MetaSR
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


conf = json.load(open("conf.json", "r", encoding="utf-8"))
predict_conf = conf["predict"]
LR_size = predict_conf["LR_size"]
k = predict_conf["k"]
inC = predict_conf["inC"]
outC = predict_conf["outC"]
r_range = predict_conf["r_range"]
r_step = predict_conf["r_step"]
use_best_model = predict_conf["use_best_model"]
FLM_use_unet = predict_conf["FLM_use_unet"]
input_img_with_original_size = predict_conf["input_img_with_original_size"]
img_dir = predict_conf["img_dir"]
use_interpolate_branch = predict_conf["use_interpolate_branch"]
r = round(float(input("input the scale factor between (%f(include), %f(exclude)), step size is %f:" % (r_range[0] + r_step, r_range[1] + r_step, r_step))), len(str(r_step).split(".")[-1]))
print("scale factor is %f" % (r,))
HR_size = (math.floor(LR_size[0] * r), math.floor(LR_size[1] * r))
transformer_to_HR = T.Resize(HR_size)
transformer_to_LR = T.Resize(LR_size)
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()
if os.path.exists("predict_result"):
    shutil.rmtree("predict_result")
os.mkdir("predict_result")


def load_one_image(img_pth):
    img = Image.open(img_pth)
    HR_img = transformer_to_HR(img)
    LR_img = transformer_to_LR(img)
    lr_img_tensor = to_tensor(LR_img)
    return lr_img_tensor, LR_img, HR_img


def load_model():
    model = MetaSR(k, inC, outC, FLM_use_unet, LR_size, use_interpolate_branch)
    model = nn.DataParallel(module=model, device_ids=[0])
    if use_best_model:
        model.load_state_dict(t.load("best.pth"))
    else:
        model.load_state_dict(t.load("epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


def predict_one_image_with_constant_lr_size(model, img_pth, result_save_dir):
    lr_img_tensor, LR_img, HR_img_gt = load_one_image(img_pth)
    lr_img_tensor = lr_img_tensor.unsqueeze(0).cuda(0)
    with t.no_grad():
        output = model(lr_img_tensor, r)[0]
    HR_img_pred = to_pil(output.cpu().detach())
    HR_img_bicubic = LR_img.resize(HR_size, resample=Image.BICUBIC)
    HR_img_pred.save(os.path.join(result_save_dir, "HR_pred.png"))
    LR_img.save(os.path.join(result_save_dir, "LR.png"))
    HR_img_gt.save(os.path.join(result_save_dir, "HR_img_gt.png"))
    HR_img_bicubic.save(os.path.join(result_save_dir, "HR_img_bicubic.png"))
    return HR_img_pred, LR_img, HR_img_gt, HR_img_bicubic


def predict_one_img_with_original_lr_size(model, img_pth, result_save_dir):
    LR_img = Image.open(img_pth)
    LR_size = LR_img.size
    HR_size = (math.floor(LR_size[0] * r), math.floor(LR_size[1] * r))
    HR_img_bicubic = LR_img.resize(HR_size, resample=Image.BICUBIC)
    LR_img_tensor = to_tensor(LR_img).unsqueeze(0)
    with t.no_grad():
        output = model(LR_img_tensor, r)[0].cpu().detach()
    HR_img_pred = to_pil(output)
    HR_img_pred.save(os.path.join(result_save_dir, "HR_pred.png"))
    LR_img.save(os.path.join(result_save_dir, "LR.png"))
    HR_img_bicubic.save(os.path.join(result_save_dir, "HR_img_bicubic.png"))
    return HR_img_pred, LR_img, None, HR_img_bicubic


def predict_one_img(model, img_pth, result_save_dir):
    if input_img_with_original_size:
        HR_img_pred, LR_img, HR_img_gt, HR_img_bicubic = predict_one_img_with_original_lr_size(model, img_pth, result_save_dir)
        return HR_img_pred, LR_img, HR_img_gt, HR_img_bicubic
    HR_img_pred, LR_img, HR_img_gt, HR_img_bicubic = predict_one_image_with_constant_lr_size(model, img_pth, result_save_dir)
    return HR_img_pred, LR_img, HR_img_gt, HR_img_bicubic


def predict(model):
    img_pths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
    for img_pth in img_pths:
        print("predict:", img_pth)
        predict_count = len(os.listdir("predict_result"))
        result_save_dir = os.path.join("predict_result", str(predict_count))
        os.makedirs(result_save_dir)
        predict_one_img(model, img_pth, result_save_dir)


if __name__ == "__main__":
    model = load_model()
    predict(model)