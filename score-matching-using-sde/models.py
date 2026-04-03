import math

import torch
import torch.nn as nn


class SinusodialPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusodialPositionEmbeddings, self).__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dim = {embedding_dim} should be divisible by 2")

        self.embedding_dim = embedding_dim

    def forward(self, ts):
        """
            timesteps : torch.tensor of size B representing the timesteps
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=ts.device) * -emb)
        # [B, 1] * [1, HD] -> [B, HD]
        emb = ts[:, None] * emb[None, :]
        # The first half is sin and second half is cos. I think this is what
        # is different when compared with attention all you need paper
        # where every even dimension was sin and odd dimension was cos.
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), axis=1)
        return emb


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.init_layer = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        """
            x: (B, D)
            t: (B, )
        """
        # t_emb = self.time_embedding(ts)
        x = self.init_layer(x)
        x = torch.cat((x, t[:, None]), axis=1)
        return self.net(x)
