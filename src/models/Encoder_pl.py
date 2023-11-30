from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import pytorch_lightning as pl

from models.common import convbnlrelu, convbnsig, convbn
from models.DKN import up_DKN
from models.ResBlock import Block, Bottleneck
from models.pl_base_model import LightningBaseModel

from config_settings import config_settings


class myself_net(LightningBaseModel):
    # def __init__(self, ResBlock: Union[Block, Bottleneck], channels_list):
    def __init__(self, settings: config_settings):
        super().__init__(settings)

        self.settings = settings

        if self.settings.res_block == "BasicBlock":
            ResBlock = Block
        elif self.settings.res_block == "Bottleneck":
            ResBlock = Bottleneck
        else:
            raise NotImplementedError

        channels_list = settings.res_channels

        self.rgb_convbn = convbn(in_channels=3, out_channels=channels_list[0] // 2, kernel_size=3, stride=1, padding=1)
        self.rgb_layer1 = self._make_layer(ResBlock, channels_list[0] // 2, channels_list[0], stride=1)
        self.rgb_layer2 = self._make_layer(ResBlock, channels_list[0] * ResBlock.expansion, channels_list[1], stride=2)
        self.rgb_layer3 = self._make_layer(ResBlock, channels_list[1] * ResBlock.expansion, channels_list[2], stride=2)
        self.rgb_layer4 = self._make_layer(ResBlock, channels_list[2] * ResBlock.expansion, channels_list[3], stride=2)
        self.rgb_layer5 = self._make_layer(ResBlock, channels_list[3] * ResBlock.expansion, channels_list[4], stride=2)

        self.dep_convbn = convbn(in_channels=1, out_channels=channels_list[0] // 2, kernel_size=3, stride=1, padding=1)
        self.dep_layer1 = self._make_layer(ResBlock, channels_list[0] // 2, channels_list[0], stride=1)
        self.dep_layer2 = self._make_layer(ResBlock, 2 * channels_list[0] * ResBlock.expansion, channels_list[1], stride=2)
        self.dep_layer3 = self._make_layer(ResBlock, 2 * channels_list[1] * ResBlock.expansion, channels_list[2], stride=2)
        self.dep_layer4 = self._make_layer(ResBlock, 2 * channels_list[2] * ResBlock.expansion, channels_list[3], stride=2)
        self.dep_layer5 = self._make_layer(ResBlock, 2 * channels_list[3] * ResBlock.expansion, channels_list[4], stride=2)

        self.dense_dep_1_16 = convbnlrelu(in_channels=2 * channels_list[4] * ResBlock.expansion, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.up_pool = nn.Upsample(scale_factor=2, mode="nearest")

        self.DKN_1_8 = up_DKN(feature_channel=channels_list[3] * ResBlock.expansion, kernel_size=3, filter_size=15, residual=True)
        self.DKN_1_4 = up_DKN(feature_channel=channels_list[2] * ResBlock.expansion, kernel_size=3, filter_size=15, residual=True)
        self.DKN_1_2 = up_DKN(feature_channel=channels_list[1] * ResBlock.expansion, kernel_size=3, filter_size=15, residual=True)
        self.DKN_1_1 = up_DKN(feature_channel=channels_list[0] * ResBlock.expansion, kernel_size=3, filter_size=15, residual=True)

    def _make_layer(self, ResBlock: Union[Block, Bottleneck], in_channels, out_channels, stride):
        return ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            i_downsample=None
            if (in_channels == out_channels and stride == 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * ResBlock.expansion),
            ),
        )

    @autocast()
    def forward(self, input):
        rgb = input["rgb"]
        sqarse_dep = input["d"]

        rgb = self.rgb_convbn(rgb)
        rgb_out1 = self.rgb_layer1(rgb)
        rgb_out2 = self.rgb_layer2(rgb_out1)
        rgb_out3 = self.rgb_layer3(rgb_out2)
        rgb_out4 = self.rgb_layer4(rgb_out3)
        rgb_out5 = self.rgb_layer5(rgb_out4)

        dep = self.dep_convbn(sqarse_dep)
        dep_out1 = self.dep_layer1(dep)
        dep_out2 = self.dep_layer2(torch.cat([dep_out1, rgb_out1], dim=1))
        dep_out3 = self.dep_layer3(torch.cat([dep_out2, rgb_out2], dim=1))
        dep_out4 = self.dep_layer4(torch.cat([dep_out3, rgb_out3], dim=1))
        dep_out5 = self.dep_layer5(torch.cat([dep_out4, rgb_out4], dim=1))

        dep_dense_1_16 = self.dense_dep_1_16(torch.cat([dep_out5, rgb_out5], dim=1))
        dep_dense_1_8 = self.DKN_1_8(self.up_pool(dep_dense_1_16), dep_out4)
        dep_dense_1_4 = self.DKN_1_4(self.up_pool(dep_dense_1_8), dep_out3)
        dep_dense_1_2 = self.DKN_1_2(self.up_pool(dep_dense_1_4), dep_out2)
        dep_dense_1_1 = self.DKN_1_1(self.up_pool(dep_dense_1_2), dep_out1)

        input["pred_full"] = dep_dense_1_1
        input["pred_1_2"] = dep_dense_1_2
        input["pred_1_4"] = dep_dense_1_4
        input["pred_1_8"] = dep_dense_1_8
        input["pred_1_16"] = dep_dense_1_16

        return input
