import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

from config_settings import settings

from models.Encoder import myself_net
from loss.netloss import NetLoss
import dataset

from tqdm import tqdm

import time, os, sys
from utility import backup_source_code, make_optimizer_scheduler


def train(model, loss, optimizer, train_loader, val_loader):
    model.train()

    train_loss = 0.0

    tepoch = tqdm(train_loader, total=len(train_loader), desc="train")
    for batch_idx, sample in enumerate(tepoch):
        optimizer.zero_grad()

        # forward
        rgb = sample["rgb"].cuda()
        sq_dep = sample["sqarse_dep"].cuda()
        gt = sample["gt"].cuda()

        pred = model(rgb, sq_dep)

        # loss
        loss_value = loss(pred, gt)
        loss_value.backward()

        # optimizer
        optimizer.step()

        # log
        train_loss += loss_value.item()
        tepoch.set_postfix({"loss": "{:.6f}".format(train_loss / (batch_idx + 1))})


def test(model, loss, test_loader):
    model.eval()

    test_loss = 0.0

    tepoch = tqdm(test_loader, total=len(test_loader), desc="test")
    for batch_idx, sample in enumerate(tepoch):
        # forward
        rgb = sample["rgb"].cuda()
        sq_dep = sample["sqarse_dep"].cuda()
        gt = sample["gt"].cuda()

        pred = model(rgb, sq_dep)

        # loss
        loss_value = loss(pred, gt)

        # log
        test_loss += loss_value.item()
        tepoch.set_postfix({"loss": "{:.6f}".format(test_loss / (batch_idx + 1))})

    return test_loss / len(test_loader)


if __name__ == "__main__":
    # 运行时间
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("current time: ", current_time)

    # 备份代码
    print("starting backup code")
    backup_source_code("../backup/{}".format(current_time))
    print("backup code to ../backup/{}".format(current_time))

    # 创建输出目录
    output_path = os.path.join("../output", current_time)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 指定设备
    assert torch.cuda.is_available(), "support only cuda"

    # dataset
    train_dataset = dataset.NYUDataset(settings.dataset_root_path, settings.train_csv, settings.sparse_density, "train")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    print("train_dataset length: ", len(train_dataset), ", train_loader length: ", len(train_loader))

    test_dataset = dataset.NYUDataset(settings.dataset_root_path, settings.test_csv, settings.sparse_density, "test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    print("test_dataset length: ", len(test_dataset), ", test_loader length: ", len(test_loader))

    # model
    if settings.model == "mini":
        net = myself_net(channels_list=[8, 16, 32, 64, 128]).cuda()
    elif settings.model == "tiny":
        net = myself_net(channels_list=[16, 32, 64, 128, 256]).cuda()
    elif settings.model == "small":
        net = myself_net(channels_list=[32, 64, 128, 256, 512]).cuda()
    elif settings.model == "base":
        net = myself_net(channels_list=[64, 128, 256, 512, 1024]).cuda()
    else:
        raise ValueError("model not found")

    # loss
    loss = NetLoss(settings.max_depth, settings.decay, settings.alpha, settings.beta).cuda()

    # optimizer & scheduler
    optimizer, scheduler = make_optimizer_scheduler(net)

    # train
    for epoch in range(1, settings.epochs + 1):
        # print lr
        for param_group in optimizer.param_groups:
            print("epoch: {}, lr: {}".format(epoch, param_group["lr"]))

            train(net, loss, optimizer, train_loader)
            loss_val = test(net, loss, test_loader)

            scheduler.step(epoch)

            torch.save(net.state_dict(), os.path.join(output_path, "epoch_%03d_net_loss_%.06f.pth" % (epoch, loss_val)))
            torch.save(optimizer.state_dict(), os.path.join(output_path, "epoch_%03d_optimizer.pth" % epoch))
