import os, sys

sys.path.append(os.path.abspath(os.path.join("../src/")))
from models.Encoder import up_DKN

from torchinfo import summary

import torch

net = up_DKN(16)
feature = torch.randn(1, 16, 228, 304)
depth = torch.randn(1, 1, 228, 304)
output = net(depth, feature)
print(
    summary(
        net,
        input_size=[(1, 16, 228, 304), (1, 1, 228, 304)],
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
    )
)
