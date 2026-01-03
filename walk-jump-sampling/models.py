"""
In this I will implement the UNet architecture. This will be used
for image denoising. The architecture is inspired from DDPM implementation
that I did in the summer. 
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, dropout_prob: float, act: nn.Module):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dropout_prob = dropout_prob
        self.act = act

        # block 1
        self.b1 = nn.Sequential(
            nn.GroupNorm(num_groups=self.groups, num_channels=self.in_channels),
            self.act,
            nn.Conv2d(self.in_channels, self.out_channels, (3, 3), 1, 1)
        )

        # block 2
        self.b2 = nn.Sequential(
            nn.GroupNorm(num_groups=self.groups, num_channels=self.out_channels),
            self.act,
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(self.out_channels, self.out_channels, (3, 3), 1, 1)
        )

        # match conv if dimension are different
        if self.in_channels != self.out_channels:
            # 1 x 1 convolution
            self.match_conv = nn.Conv2d(self.in_channels, self.out_channels, 1)
        else:
            self.match_conv = nn.Identity()


    def forward(self, x):
        # pass to block 1
        h = self.b1(x)

        # pass to block 2
        h = self.b2(h)

        return h + self.match_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.in_channels = in_channels
        # kernel size = 3, stride = 2, padding = 1
        self.conv = nn.Conv2d(self.in_channels, self.in_channels, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class UNetSingleNoiseScale(nn.Module):
    def __init__(self, channels, dim, norm_groups, dropout_prob=0):
        super(UNetSingleNoiseScale, self).__init__()

        self.channels = channels
        self.dim = dim
        self.norm_groups = norm_groups
        self.dropout_prob = dropout_prob

        # activation
        self.act = nn.SiLU()

        # initial conv
        self.init_conv = nn.Conv2d(self.channels, self.dim, (3, 3), 1, 1)

        # downsampling or encoder layers
        self.res1 = nn.Sequential(
            ResidualBlock(self.dim, self.dim, self.norm_groups, self.dropout_prob, self.act),
            ResidualBlock(self.dim, self.dim, self.norm_groups, self.dropout_prob, self.act)
        )
        self.ds1 = DownSample(self.dim)

        self.res2 = nn.Sequential(
            ResidualBlock(self.dim, 2 * self.dim, self.norm_groups, self.dropout_prob, self.act),
            ResidualBlock(2 * self.dim, 2 * self.dim, self.norm_groups, self.dropout_prob, self.act)
        )
        self.ds2 = DownSample(2 * self.dim)

        # middle block
        self.res3 = nn.Sequential(
            ResidualBlock(2 * self.dim, 2 * self.dim, self.norm_groups, self.dropout_prob, self.act),
            ResidualBlock(2 * self.dim, 2 * self.dim, self.norm_groups, self.dropout_prob, self.act)
        )
        self.res4 = nn.Sequential(
            ResidualBlock(2 * self.dim, 4 * self.dim, self.norm_groups, self.dropout_prob, self.act),
            ResidualBlock(4 * self.dim, 4 * self.dim, self.norm_groups, self.dropout_prob, self.act),
            ResidualBlock(4 * self.dim, 2 * self.dim, self.norm_groups, self.dropout_prob, self.act)
        )

        # upsampling
        self.us1 = UpSample(2 * self.dim)
        self.res5 = nn.Sequential(
            ResidualBlock(4 * self.dim, 2 * self.dim, self.norm_groups, self.dropout_prob, self.act),
            ResidualBlock(2 * self.dim, self.dim, self.norm_groups, self.dropout_prob, self.act)
        )

        self.us2 = UpSample(self.dim)
        self.res6 = nn.Sequential(
            ResidualBlock(2 * self.dim, self.dim, self.norm_groups, self.dropout_prob, self.act),
            ResidualBlock(self.dim, self.dim, self.norm_groups, self.dropout_prob, self.act)
        )

        # end
        self.end = nn.Sequential(
            nn.GroupNorm(self.norm_groups, self.dim),
            self.act,
            nn.Conv2d(self.dim, self.channels, 3, 1, 1)
        )

    def forward(self, x):
        h = self.init_conv(x)

        # downsampling
        h1 = self.res1(h)
        h2 = self.ds1(h1)
        h3 = self.res2(h2)
        h4 = self.ds2(h3)

        # middle
        h4 = self.res3(h4)
        h5 = self.res4(h4)

        # upsampling
        h5 = self.us1(h5)
        h5 = torch.cat((h5, h3), dim=1)
        h6 = self.res5(h5)
        h7 = self.us2(h6)
        h7 = torch.cat((h7, h1), dim=1)
        h8 = self.res6(h7)

        # end
        out = self.end(h8)
        return out
