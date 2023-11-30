import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profilers import SimpleProfiler

from config_settings import settings
from models.Encoder_pl import myself_net
from loss.netloss import NetLoss
from dataset import build_dataloader
from utility import backup_source_code, make_optimizer_scheduler

from tqdm import tqdm
from easydict import EasyDict
import time, os, sys

if __name__ == "__main__":
    # 进程的GPU可见性，仅使用指定的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpus

    # 随机性
    pl.seed_everything(settings.seed)

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(settings.tb_log_folder, name="myself", default_hp_metric=False)

    # profiler
    profiler = SimpleProfiler(filename="profiler")

    # data loader
    train_loader, val_loader, test_loader = build_dataloader(settings)

    # model
    model = myself_net(settings)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="../checkpoint",
        filename="myself-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[i for i in range(settings.num_gpus)],
        strategy="ddp",
        max_epochs=settings.epochs,
        limit_train_batches=0.6,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val/rmse", patience=5, verbose=True, mode="min"),
        ],
        profiler=profiler,
        logger=tb_logger,
        log_every_n_steps=settings.log_freq,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        precision=16, # 半精度
        # num_nodes=1,
        # distributed_backend="ddp",
        # resume_from_checkpoint="../checkpoint/myself-epoch=02-val_loss=0.00.ckpt",
    )

    # train
    print("start training...")
    trainer.fit(model, train_loader, val_loader)
