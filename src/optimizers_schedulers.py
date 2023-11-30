import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from config_settings import config_settings


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


def make_optimizer_scheduler(settings: config_settings, params):
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

    optimizer = optimizer_class(params, **kwargs_optimizer)

    # scheduler
    if settings.scheduler == "LambdaLR":
        # 使用自定义的学习率衰减策略
        decay = settings.LambdaLR_decay
        gamma = settings.LambdaLR_gamma
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
    elif settings.scheduler == "CosineAnnealingLR":
        # 余弦退火
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=settings.epochs - 4, eta_min=1e-5)
    else:
        raise NotImplementedError

    return optimizer, scheduler
