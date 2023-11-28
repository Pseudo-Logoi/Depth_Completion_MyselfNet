import torch
import torch.nn as nn

import pytorch_lightning as pl

from config_settings import settings


def train(model, loss, optimizer, scheduler, train_loader, val_loader, epochs, device):
    pass


def test(model, loss, test_loader, device):
    pass


if __name__ == "__main__":
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
