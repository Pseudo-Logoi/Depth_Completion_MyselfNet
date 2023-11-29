import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from models.common import convbnlrelu, convbnsig, convbn


def grid_generator(k, r, n):
    """grid_generator
    Parameters
    ---------
    k : filter_size, int
    r: kernel_size, int
    n: number of grid, int
    Returns
    -------
    torch.Tensor.shape = (n, r, r, 2)
    """
    # 生成 r x r 的网格
    grid_x, grid_y = torch.meshgrid(
        [
            torch.linspace(k // 2, k // 2 + r - 1, steps=r),  # 从 k//2 到 k//2+r-1, 生成 r 个数
            torch.linspace(k // 2, k // 2 + r - 1, steps=r),  # 从 k//2 到 k//2+r-1, 生成 r 个数
        ],
        indexing="ij",  # ij, xy, meshgrid 的 indexing
    )
    grid = torch.stack([grid_x, grid_y], 2).view(r, r, 2)
    return grid.unsqueeze(0).repeat(n, 1, 1, 1)


class up_DKN(nn.Module):
    def __init__(self, feature_channel, kernel_size=3, filter_size=15, residual=True) -> None:
        super().__init__()

        self.residual = residual
        self.filter_size = filter_size
        self.kernel_size = kernel_size

        self.depth_guided_convbnlrelu = convbnlrelu(in_channels=1, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)

        self.weight_convbn = convbn(in_channels=2 * feature_channel, out_channels=kernel_size**2, kernel_size=3, stride=1, padding=1)
        self.offset_convbn = convbn(in_channels=2 * feature_channel, out_channels=2 * kernel_size**2, kernel_size=3, stride=1, padding=1)

    def DKN_Interpolate(self, depth, weight, offset):
        """
        depth shape: (b, 1, h, w)
        weight shape: (b, r^2, h, w)
        offset shape: (b, 2*r^2, h, w)
        """

        if self.residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)  # 行标准化，每个值减去每行的均值
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)  # 行归一化，每个值除以每行的和

        b, h, w = depth.size(0), depth.size(2), depth.size(3)
        k = self.filter_size
        r = self.kernel_size
        hw = h * w

        # (b, 2*r^2, h, w) -> (b, h, w, 2*r^2) -> (b*hw, r, r, 2)
        offset = offset.permute(0, 2, 3, 1).contiguous().view(b * hw, r, r, 2)
        # (b, r^2, h, w) -> (b, h, w, r^2) -> (b*hw, r^2, 1)
        weight = weight.permute(0, 2, 3, 1).contiguous().view(b * hw, r * r, 1)

        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b * hw).cuda()

        coord = grid + offset
        coord = (coord / k * 2) - 1

        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(depth, k, padding=k // 2).permute(0, 2, 1).contiguous().view(b * hw, 1, k, k)

        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(depth_col.float(), coord, align_corners=True).view(b * hw, 1, -1)

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h,w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h, w).float()

        if self.residual:
            out += depth.float()

        return out

    @autocast()
    def forward(self, depth, feature):
        depth_feature = self.depth_guided_convbnlrelu(depth)
        weight = torch.sigmoid(self.weight_convbn(torch.cat([depth_feature, feature], dim=1)))
        offset = self.offset_convbn(torch.cat([depth_feature, feature], dim=1))

        return self.DKN_Interpolate(depth, weight, offset)
