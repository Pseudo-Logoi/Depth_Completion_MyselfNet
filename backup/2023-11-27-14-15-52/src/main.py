import torch
import torch.nn as nn

import pytorch_lightning as pl

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

    # loss

    # optimizer

    # scheduler

    # trainer

    # train

    # test
