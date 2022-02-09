import torch as t
from model import MetaSR
from torch import nn, optim
from dataset import make_loader, get_collate
import os
import json
from loss import CostumLoss


conf = json.load(open("conf.json", "r", encoding="utf-8"))
train_conf = conf["train"]
CUDA_VISIBLE_DEVICES = train_conf["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
epoch = train_conf["epoch"]
batch_size = train_conf["batch_size"]
init_lr = train_conf["init_lr"]
final_lr = train_conf["final_lr"]
lr_de_rate = (final_lr / init_lr) ** (1 / epoch)
LR_size = train_conf["LR_size"]
train_hr_img_dir = train_conf["train_hr_img_dir"]
valid_hr_img_dir = train_conf["valid_hr_img_dir"]
inC = train_conf["inC"]
outC = train_conf["outC"]
k = train_conf["k"]
print_step = train_conf["print_step"]
r_range = train_conf["r_range"]
r_step = train_conf["r_step"]
FLM_use_unet = train_conf["FLM_use_unet"]
num_workers = train_conf["num_workers"]
use_costum_loss = train_conf["use_costum_loss"]
weight_decay = train_conf["weight_decay"]
use_interpolate_branch = train_conf["use_interpolate_branch"]
device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
best_valid_loss = float("inf")
collate_fn = get_collate(r_range, r_step)


def train_epoch(current_epoch, epoch, model, criterion, optimizer, train_loader):
    model.train()
    step = len(train_loader)
    current_step = 1
    for d_train, l_train, r in train_loader:
        d_train = t.cat(d_train, dim=0)
        l_train = t.cat(l_train, dim=0)
        d_train_cuda = d_train.cuda(device_ids[0])
        l_train_cuda = l_train.cuda(device_ids[0])
        train_output = model(d_train_cuda, r)
        train_loss = criterion(train_output, l_train_cuda)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if current_step % print_step == 0:
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f, r:%.5f" % (current_epoch, epoch, current_step, step, train_loss.item(), r))
        current_step += 1
    print("saving epoch model......")
    t.save(model.state_dict(), "epoch.pth")
    return model


def valid_epoch(model, current_epoch, criterion, valid_loader):
    global best_valid_loss
    model.eval()
    step = len(valid_loader)
    accum_loss = 0
    for d_valid, l_valid, r in valid_loader:
        d_valid = t.cat(d_valid, dim=0)
        l_valid = t.cat(l_valid, dim=0)
        d_valid_cuda = d_valid.cuda(device_ids[0])
        l_valid_cuda = l_valid.cuda(device_ids[0])
        with t.no_grad():
            valid_output = model(d_valid_cuda, r)
            valid_loss = criterion(valid_output, l_valid_cuda)
            accum_loss += valid_loss.item()
    avg_loss = accum_loss / step
    print("###########validation epoch:%d##############" % (current_epoch,))
    if avg_loss < best_valid_loss:
        print("saving best model......")
        best_valid_loss = avg_loss
        t.save(model.state_dict(), "best.pth")
    print("valid_loss:%.5f" % (avg_loss,))
    print("############################################")
    return model


def main():
    model = MetaSR(k=k, inC=inC, outC=outC, FLM_use_unet=FLM_use_unet, LR_size=LR_size, use_interpolate_branch=use_interpolate_branch)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    if not use_costum_loss:
        criterion = nn.L1Loss().cuda(device_ids[0])
    else:
        criterion = CostumLoss().cuda(device_ids[0])
    optimizer = optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    lr_sch = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=lr_de_rate)
    for e in range(epoch):
        print("lr:%f" % (lr_sch.get_lr()[0],))
        train_loader = make_loader(LR_size, train_hr_img_dir, batch_size, collate_fn, num_workers)
        valid_loader = make_loader(LR_size, valid_hr_img_dir, batch_size, collate_fn, num_workers)
        model = train_epoch(e + 1, epoch, model, criterion, optimizer, train_loader)
        model = valid_epoch(model, e + 1, criterion, valid_loader)
        lr_sch.step()


if __name__ == "__main__":
    main()

