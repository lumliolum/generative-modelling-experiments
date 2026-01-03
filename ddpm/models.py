"""
in this we will try to implement the U-NET architecture given in the paper.
reference taken from 
- original code : https://github.com/hojonathanho/diffusion/tree/master
- annotated diffusion : https://huggingface.co/blog/annotated-diffusion
"""

import math
from typing import List

import torch
import torch.nn as nn
from einops import rearrange, einsum


class SinusodialPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusodialPositionEmbeddings, self).__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dim = {embedding_dim} should be divisible by 2")

        self.embedding_dim = embedding_dim

    def forward(self, time):
        """
            time : torch.tensor of size B representing the timesteps
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb)
        # [B, 1] * [1, HD] -> [B, HD]
        emb = time[:, None] * emb[None, :]
        # The first half is sin and second half is cos. I think this is what
        # is different when compared with attention all you need paper
        # where every even dimension was sin and odd dimension was cos.
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), axis=1)
        return emb


class ResNetBlock(nn.Module):
    """
        See appendix B of the paper.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim, groups, dropout_prob):
        super(ResNetBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.groups = groups
        self.dropout_prob = dropout_prob

        # in the original codebase, authors are doing normalization first
        # and then apply convolution layer
        # block 1
        self.b1 = nn.Sequential(
            nn.GroupNorm(num_groups=self.groups, num_channels=self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, (3, 3), 1, 1),
        )

        # after b1, we will add time embeddings. Similar to above
        # in the original code, they are applying nonlinearity first
        # and then linear layers
        self.time_embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.out_channels)
        )

        # block 2
        self.b2 = nn.Sequential(
            nn.GroupNorm(num_groups=self.groups, num_channels=self.out_channels),
            nn.SiLU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(self.out_channels, self.out_channels, (3, 3), 1, 1)
        )

        # match conv if dimension are different
        if self.in_channels != self.out_channels:
            # 1 x 1 convolution
            self.match_conv = nn.Conv2d(self.in_channels, self.out_channels, 1)
        else:
            self.match_conv = nn.Identity()

    def forward(self, x, time_embed):
        # pass to block 1
        h = self.b1(x)

        # add time embedding. The time embedding is added to each pixel
        h = h + self.time_embed_proj(time_embed)[:, :, None, None]

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


class Attention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(Attention, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_channels = self.in_channels // self.num_heads
        # 1 x 1 convolution
        # in the original code, authors had bias = True for q, k, v. don't know any particular reason for it.
        self.q = nn.Conv2d(self.in_channels, self.in_channels, 1, 1, bias=False)
        self.k = nn.Conv2d(self.in_channels, self.in_channels, 1, 1, bias=False)
        self.v = nn.Conv2d(self.in_channels, self.in_channels, 1, 1, bias=False)
        # out
        self.o = nn.Conv2d(self.in_channels, self.in_channels, 1, 1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape

        # x -> (B, C, H, H)
        q, k, v = self.q(x), self.k(x), self.v(x)
        # (B, attention heads, dimension, sequence length)
        qhs = rearrange(q, 'b (h1 c) h w -> b h1 c (h w)', h1=self.num_heads)
        khs = rearrange(k, 'b (h1 c) h w -> b h1 c (h w)', h1=self.num_heads)
        vhs = rearrange(v, 'b (h1 c) h w -> b h1 c (h w)', h1=self.num_heads)

        # (b, n, s, d) x (b, n, d, s) -> (b, n, s, s)
        sim = einsum(qhs, khs, 'b n d s1, b n d s2 -> b n s1 s2')
        # scale
        sim = sim / math.sqrt(self.head_channels)
        # softmax
        sim = torch.softmax(sim, -1)

        # (b, n, s1, s) (b, n, d, s) -> (b, n, s1, d)
        attn = einsum(sim, vhs, 'b n s1 s, b n d s -> b n s1 d')
        out = rearrange(attn, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        out = self.o(out)
        return out


class UNet(nn.Module):
    def __init__(self, channels, dim, dim_multiplier: List, num_blocks_per_resolution, res_groups, num_heads, dropout_prob=0):
        super(UNet, self).__init__()
        self.channels = channels
        # represents the hidden channels in the network
        self.dim = dim
        # represents the multiplier for hidden channels as we go down the network
        self.dim_multiplier = dim_multiplier
        # number of residual block per resolution.
        self.num_blocks_per_resolution = num_blocks_per_resolution
        # group norm groups in residua block
        self.res_groups = res_groups
        # number of attention heads
        self.num_heads = num_heads
        # dropout_prob
        self.dropout_prob = dropout_prob

        # calculate the channels at each level
        dims = [self.dim] + [self.dim * m for m in self.dim_multiplier]

        self.time_embed_dim = self.dim * 4

        # timestep embedding
        self.time_embed = nn.Sequential(
            SinusodialPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        # initial conv
        self.init_conv = nn.Conv2d(self.channels, self.dim, (3, 3), 1, 1)

        # downsampling
        self.downs = nn.ModuleList()

        for level in range(len(self.dim_multiplier)):
            # here we fix the resolution. As per the paper, there are
            # two residual block per resolution.
            for block in range(self.num_blocks_per_resolution):
                if block == 0:
                    in_dim, out_dim = dims[level], dims[level + 1]
                else:
                    in_dim, out_dim = dims[level + 1], dims[level + 1]

                # in the original paper, attention block was present
                # only for resolution = 16. Here I am adding for everyone
                self.downs.append(
                    nn.ModuleList(
                        [
                            ResNetBlock(
                                in_channels=in_dim,
                                out_channels=out_dim,
                                time_embed_dim=self.time_embed_dim,
                                groups=self.res_groups,
                                dropout_prob=self.dropout_prob
                            ),
                            Attention(
                                in_channels=out_dim,
                                num_heads=self.num_heads
                            ),
                        ]
                    )
                )

            # as all the required residual blocks are added, we will add
            # downsample block now
            if level < len(self.dim_multiplier) - 1:
                self.downs.append(
                    # add the downsample block
                    DownSample(in_channels=dims[level + 1])
                )

        # middle
        self.middle = nn.ModuleList(
            [
                ResNetBlock(
                    in_channels=dims[-1],
                    out_channels=dims[-1],
                    time_embed_dim=self.time_embed_dim,
                    groups=self.res_groups,
                    dropout_prob=self.dropout_prob
                ),
                Attention(
                    in_channels=dims[-1],
                    num_heads=self.num_heads
                ),
                ResNetBlock(
                    in_channels=dims[-1],
                    out_channels=dims[-1],
                    time_embed_dim=self.time_embed_dim,
                    groups=self.res_groups,
                    dropout_prob=self.dropout_prob
                )
            ]
        )

        # upsampling
        self.ups = nn.ModuleList()
        for level in range(len(dim_multiplier)):
            for block in range(self.num_blocks_per_resolution):
                if level == 0:
                    in_dim = 2 * dims[-(level + 1)]
                elif block == 0:
                    in_dim = dims[-(level + 1)] + dims[-level]
                else:
                    in_dim = 2 * dims[-(level + 1)]

                self.ups.append(
                    nn.ModuleList(
                        [
                            ResNetBlock(
                                in_channels=in_dim,
                                out_channels=dims[-(level + 1)],
                                time_embed_dim=self.time_embed_dim,
                                groups=self.res_groups,
                                dropout_prob=self.dropout_prob
                            ),
                            Attention(
                                in_channels=dims[-(level + 1)],
                                num_heads=self.num_heads
                            )
                        ]
                    )
                )

            # add upsampling block
            if level != len(dim_multiplier) - 1:
                self.ups.append(
                    UpSample(dims[-(level + 1)])
                )

        # end
        self.end = nn.Sequential(
            nn.GroupNorm(self.res_groups, self.dim),
            nn.SiLU(),
            nn.Conv2d(self.dim, self.channels, 3, 1, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)
        # time embedding
        te = self.time_embed(time)

        # save the values for upsampling
        h = []
        index = 0
        for _ in range(len(self.dim_multiplier)):
            for _ in range(self.num_blocks_per_resolution):
                # resnet block
                x = self.downs[index][0](x, te)
                # attention
                x = self.downs[index][1](x)
                h.append(x)
                index += 1

            # downsample (it will not be there at the end)
            if index < len(self.downs) and isinstance(self.downs[index], DownSample):
                x = self.downs[index](x)
                index += 1

        # middle block
        x = self.middle[0](x, te)
        x = self.middle[1](x)
        x = self.middle[2](x, te)

        # upsampling
        index = 0
        for _ in range(len(self.dim_multiplier)):
            for _ in range(self.num_blocks_per_resolution):
                # encoder decoder connections
                x = torch.cat((x, h.pop()), dim=1)
                # resnet block
                x = self.ups[index][0](x, te)
                # attention
                x = self.ups[index][1](x)
                index += 1

            # upsample
            if index < len(self.ups) and isinstance(self.ups[index], UpSample):
                x = self.ups[index](x)
                index += 1

        # end
        x = self.end(x)
        return x


if __name__ == "__main__":
    model = UNet(
        channels=3,
        dim=128,
        dim_multiplier=(1, 2, 2, 4),
        num_blocks_per_resolution=2,
        res_groups=32,
        num_heads=1,
        dropout_prob=0.1
    )
    x = torch.randn((16, 3, 32, 32))
    t = torch.randint(low=0, high=100, size=(16, ), dtype=torch.long)
    print(model(x, t).shape)

    # calculate the parameters of the model
    tp = 0
    for p in model.parameters():
        tp += torch.numel(p)

    # in M
    print(tp / 1000000)
