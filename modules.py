import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Optional
from math import pi


class CoordinateEncoding(nn.Module):
    def __init__(self, proj_matrix):
        super().__init__()
        self.register_buffer('proj_matrix', proj_matrix)
        self.in_dim = self.proj_matrix.size(0)
        self.out_dim = self.proj_matrix.size(1) * 2

    def forward(self, x):
        shape = x.shape
        channels = shape[-1]

        assert channels == self.in_dim, f'Expected input to have {self.in_dim} channels (got {channels} channels)'

        x = x.reshape(-1, channels)
        x = x @ self.proj_matrix

        x = x.view(*shape[:-1], -1)
        x = 2 * pi * x

        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class IdentityPositionalEncoding(CoordinateEncoding):
    def __init__(self, in_dim):
        super().__init__(torch.eye(in_dim))
        self.out_dim = in_dim

    def forward(self, x):
        return x


class GaussianFourierFeatureTransform(CoordinateEncoding):
    def __init__(self, in_dim: int, mapping_size: int = 32, scale: float = 1.0, seed=None):
        super().__init__(self.get_transform_matrix(in_dim, mapping_size, scale, seed=seed))
        self.mapping_size = mapping_size
        self.scale = scale
        self.seed = seed

    @classmethod
    def get_transform_matrix(cls, in_dim, mapping_size, scale, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        return torch.randn((in_dim, mapping_size), generator=generator) * scale

    @classmethod
    def from_matrix(cls, projection_matrix):
        in_dim, mapping_size = projection_matrix.shape
        feature_transform = cls(in_dim, mapping_size)
        feature_transform.projection_matrix.data = projection_matrix
        return feature_transform


class NeRFPositionalEncoding(CoordinateEncoding):
    def __init__(self, in_dim, n=10):
        super().__init__((2.0 ** torch.arange(n))[None, :])
        self.out_dim = n * 2 * in_dim

    def forward(self, x):
        shape = x.shape
        x = x.unsqueeze(-1) * self.proj_matrix
        x = pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*shape[:-1], -1)
        return x


class LinearBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 linear_cls,
                 activation=nn.ReLU,
                 bias=True,
                 is_first=False,
                 is_last=False):
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        self.linear = linear_cls(in_features, out_features, bias=bias)
        self.bias = bias
        self.is_first = is_first
        self.is_last = is_last
        self.activation = None if is_last else activation()

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

    def __repr__(self):
        return f'LinearBlock(in_features={self.in_f}, out_features={self.out_f}, linear_cls={self.linear}, ' \
               f'activation={self.activation}, bias={self.bias}, is_first={self.is_first}, is_last={self.is_last})'


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLinear(LinearBlock):
    def __init__(self, in_features, out_features, linear_cls=nn.Linear, w0=30, bias=True, is_first=False, is_last=False):
        super().__init__(in_features, out_features, linear_cls, partial(Sine, w0), bias, is_first, is_last)
        self.w0 = w0
        self.init_weights()

    def init_weights(self):
        if self.is_first:
            b = 1 / self.in_f
        else:
            b = np.sqrt(6 / self.in_f) / self.w0

        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)


class BaseBlockFactory:
    def __call__(self, in_f, out_f, is_first=False, is_last=False):
        raise NotImplementedError


class LinearBlockFactory(BaseBlockFactory):
    def __init__(self, linear_cls=nn.Linear, activation_cls=nn.ReLU, bias=True):
        self.linear_cls = linear_cls
        self.activation_cls = activation_cls
        self.bias = bias

    def __call__(self, in_f, out_f, is_first=False, is_last=False):
        return LinearBlock(in_f, out_f, self.linear_cls, self.activation_cls, self.bias, is_first, is_last)


class SirenBlockFactory(BaseBlockFactory):
    def __init__(self, linear_cls=nn.Linear, w0=30, bias=True):
        self.linear_cls = linear_cls
        self.w0 = w0
        self.bias = bias

    def __call__(self, in_f, out_f, is_first=False, is_last=False):
        return SirenLinear(in_f, out_f, self.linear_cls, self.w0, self.bias, is_first, is_last)


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 block_factory: BaseBlockFactory,
                 dropout: float = 0.0,
                 residual_connections: bool = False,
                 final_activation: Optional[str] = None):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual_connections = residual_connections

        self.blocks = nn.ModuleList()

        if self.num_layers < 1:
            raise ValueError(f'num_layers must be >= 1 (input to output); got {self.num_layers}')

        for i in range(self.num_layers):
            in_feat = self.in_dim if i == 0 else self.hidden_dim
            out_feat = self.out_dim if i + 1 == self.num_layers else self.hidden_dim

            is_first = i == 0
            is_last = i + 1 == self.num_layers

            self.blocks.append(block_factory(
                in_feat,
                out_feat,
                is_first=is_first,
                is_last=is_last
            ))

            if not is_last and dropout:
                self.blocks.append(nn.Dropout(dropout))

        self.final_activation = final_activation
        if final_activation is None:
            self.final_activation = nn.Identity()

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if i + 1 < len(self.blocks) and self.residual_connections:
                x = block(x) + x
            else:
                x = block(x)
        return self.final_activation(x)


class GON(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 input_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 block_factory: BaseBlockFactory,
                 dropout: float = 0.0,
                 learn_origin=False,
                 final_activation=None):
        super().__init__()

        self.latent_dim = latent_dim

        if learn_origin:
            self.origin = nn.Parameter(torch.zeros(1, latent_dim))
        else:
            self.register_buffer('origin', torch.zeros(1, latent_dim))

        self.net = MLP(
            in_dim=latent_dim + input_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            block_factory=block_factory,
            dropout=dropout,
            final_activation=final_activation
        )

    def get_origin_latent(self, n):
        return self.origin.repeat(n, 1)

    def loss_inner(self, output, target):
        return F.binary_cross_entropy(
            output.view(-1), target.view(-1), reduction='none'
        ).view(target.shape).sum(0).mean()

    def loss_outer(self, output, target):
        return F.binary_cross_entropy(
            output.view(-1), target.view(-1), reduction='none'
        ).view(target.shape).mean()

    def infer_latents(self, data, model_input):
        origin = self.get_origin_latent(len(data)).requires_grad_(True)
        out = self(model_input, origin)
        inner_loss = self.loss_inner(out, data)
        latent = origin - torch.autograd.grad(inner_loss, [origin], create_graph=True, retain_graph=True)[0]
        return latent, inner_loss

    def forward(self, input, latent):
        b, w, h, c = input.shape
        latent = latent[:, None, None, :].repeat(1, w, h, 1)
        x = torch.cat([latent, input], dim=-1)
        x = self.net(x)
        return torch.sigmoid(x)
