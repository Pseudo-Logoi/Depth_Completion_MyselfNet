import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

from models.Encoder import myself_net
from loss.netloss import NetLoss
import dataset

from config_settings import settings

import time
from utility import backup_source_code, make_optimizer_scheduler


def train(model, loss, optimizer, scheduler, train_loader, val_loader, epochs):
    pass


def test(model, loss, test_loader):
    pass


if __name__ == "__main__":
    # 运行时间
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("current time: ", current_time)

    # 备份代码
    print("starting backup code")
    backup_source_code("../backup/{}".format(current_time))
    print("backup code to ../backup/{}".format(current_time))

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
    train(net, loss, optimizer, scheduler, train_loader, None, settings.epochs)

    # test
