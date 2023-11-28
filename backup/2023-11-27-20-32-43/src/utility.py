"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    Some of useful functions are defined here.
"""


import os
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from config_settings import settings


class LRFactor:
    def __init__(self, decay, gamma):
        assert len(decay) == len(gamma)

        self.decay = decay
        self.gamma = gamma

    def get_factor(self, epoch):
        for d, g in zip(self.decay, self.gamma):
            if epoch < d:
                return g
        return self.gamma[-1]


def convert_str_to_num(val, t):
    val = val.replace("'", "")
    val = val.replace('"', "")

    if t == "int":
        val = [int(v) for v in val.split(",")]
    elif t == "float":
        val = [float(v) for v in val.split(",")]
    else:
        raise NotImplementedError

    return val


def make_optimizer_scheduler(target):
    # optimizer
    kwargs_optimizer = {"lr": settings.lr, "weight_decay": settings.weight_decay}
    if settings.optimizer == "SGD":
        optimizer_class = optim.SGD
        kwargs_optimizer["momentum"] = settings.momentum
    elif settings.optimizer == "ADAM":
        optimizer_class = optim.Adam
        kwargs_optimizer["betas"] = settings.betas
        kwargs_optimizer["eps"] = settings.epsilon
    elif settings.optimizer == "RMSprop":
        optimizer_class = optim.RMSprop
        kwargs_optimizer["eps"] = settings.epsilon
    else:
        raise NotImplementedError

    trainable = target.param_groups if hasattr(target, "param_groups") else filter(lambda x: x.requires_grad, target.parameters())
    optimizer = optimizer_class(trainable, **kwargs_optimizer)

    # scheduler
    if settings.scheduler == "LambdaLR":
        # 使用自定义的学习率衰减策略
        decay = convert_str_to_num(settings.LambdaLR_decay, "int")
        gamma = convert_str_to_num(settings.LambdaLR_gamma, "float")
        assert len(decay) == len(gamma), "decay and gamma must have same length"
        calculator = LRFactor(decay, gamma)
        scheduler = lrs.LambdaLR(optimizer, calculator.get_factor)
    elif settings.scheduler == "MultiStepLR":
        # 分阶段(milestones)衰减(gamma)学习率
        scheduler = lrs.MultiStepLR(optimizer, milestones=settings.MultiStepLR_milestones, gamma=settings.MultiStepLR_gamma)
    elif settings.scheduler == "ReduceLROnPlateau":
        # 当指标不再变化时，降低学习率
        scheduler = lrs.ReduceLROnPlateau(
            optimizer, mode="min", factor=settings.ReduceLROnPlateau_factor, patience=settings.ReduceLROnPlateau_patience, verbose=True
        )

    return optimizer, scheduler


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*", "data", "backup")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree("..", backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


if __name__ == "__main__":
    print(convert_str_to_num("1,2,3", "int"))  # [1, 2, 3]
    backup_source_code("../backup/test_backup")
