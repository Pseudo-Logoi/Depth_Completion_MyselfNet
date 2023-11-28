import torch
import torch.nn as nn
import torch.nn.functional as F


def convbnlrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=False),
    )


def convbnsig(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid(),
    )


def convbn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(out_channels)
    )


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.batch_norm1(self.conv1(x))
        x = self.relu(x)
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity

        x = self.relu(x)
        return x


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

    def forward(self, depth, feature):
        depth_feature = self.depth_guided_convbnlrelu(depth)
        weight = torch.sigmoid(self.weight_convbn(torch.cat([depth_feature, feature], dim=1)))
        offset = self.offset_convbn(torch.cat([depth_feature, feature], dim=1))

        return self.DKN_Interpolate(depth, weight, offset)


class myself_net(nn.Module):
    def __init__(self, channels_list):
        super().__init__()

        self.channels_list = channels_list

        self.rgb_convbn = convbn(in_channels=3, out_channels=channels_list[0] // 2, kernel_size=3, stride=1, padding=1)
        self.rgb_layer1 = self._make_layer(Block, channels_list[0] // 2, channels_list[0], stride=1)
        self.rgb_layer2 = self._make_layer(Block, channels_list[0], channels_list[1], stride=2)
        self.rgb_layer3 = self._make_layer(Block, channels_list[1], channels_list[2], stride=2)
        self.rgb_layer4 = self._make_layer(Block, channels_list[2], channels_list[3], stride=2)
        self.rgb_layer5 = self._make_layer(Block, channels_list[3], channels_list[4], stride=2)

        self.dep_convbn = convbn(in_channels=1, out_channels=channels_list[0] // 2, kernel_size=3, stride=1, padding=1)
        self.dep_layer1 = self._make_layer(Block, channels_list[0] // 2, channels_list[0], stride=1)
        self.dep_layer2 = self._make_layer(Block, 2 * channels_list[0], channels_list[1], stride=2)
        self.dep_layer3 = self._make_layer(Block, 2 * channels_list[1], channels_list[2], stride=2)
        self.dep_layer4 = self._make_layer(Block, 2 * channels_list[2], channels_list[3], stride=2)
        self.dep_layer5 = self._make_layer(Block, 2 * channels_list[3], channels_list[4], stride=2)

        self.dense_dep_1_16 = convbnlrelu(in_channels=2 * channels_list[4], out_channels=1, kernel_size=1, stride=1, padding=0)

        self.up_pool = nn.Upsample(scale_factor=2, mode="nearest")

        self.DKN_1_8 = up_DKN(feature_channel=channels_list[3], kernel_size=3, filter_size=15, residual=True)
        self.DKN_1_4 = up_DKN(feature_channel=channels_list[2], kernel_size=3, filter_size=15, residual=True)
        self.DKN_1_2 = up_DKN(feature_channel=channels_list[1], kernel_size=3, filter_size=15, residual=True)
        self.DKN_1_1 = up_DKN(feature_channel=channels_list[0], kernel_size=3, filter_size=15, residual=True)

    def _make_layer(self, ResBlock, in_channels, out_channels, stride):
        return ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            i_downsample=None
            if (in_channels == out_channels and stride == 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            ),
        )

    def forward(self, rgb, sqarse_dep):
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
        return dep_dense_1_1, dep_dense_1_2, dep_dense_1_4, dep_dense_1_8, dep_dense_1_16
