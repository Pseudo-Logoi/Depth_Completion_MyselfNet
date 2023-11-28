import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

from models.Encoder import myself_net

from config_settings import settings

import time
from utility import backup_source_code


def train(model, loss, optimizer, scheduler, train_loader, val_loader, epochs, device):
    pass


def test(model, loss, test_loader, device):
    pass


if __name__ == "__main__":
    # 运行时间
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("current time: ", current_time)

    # 备份代码
    backup_source_code("../backup/{}".format(current_time))

    # 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device("cuda"), "support only cuda"

    # dataset

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

    # optimizer

    # scheduler

    # trainer

    # train

    # test
