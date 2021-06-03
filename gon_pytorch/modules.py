import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Optional, List, Callable
from math import pi, sqrt


class CoordinateEncoding(nn.Module):
    def __init__(self, proj_matrix, is_trainable=False):
        super().__init__()
        if is_trainable:
            self.register_parameter('proj_matrix', nn.Parameter(proj_matrix))
        else:
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
    def __init__(self, in_dim: int, mapping_size: int = 32, sigma: float = 1.0, seed=None):
        super().__init__(self.get_transform_matrix(in_dim, mapping_size, sigma, seed=seed))
        self.mapping_size = mapping_size
        self.sigma = sigma
        self.seed = seed

    @classmethod
    def get_transform_matrix(cls, in_dim, mapping_size, sigma, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        return torch.normal(mean=0, std=sigma, size=(in_dim, mapping_size), generator=generator)

    @classmethod
    def from_matrix(cls, projection_matrix):
        in_dim, mapping_size = projection_matrix.shape
        feature_transform = cls(in_dim, mapping_size)
        feature_transform.projection_matrix.data = projection_matrix
        return feature_transform

    def __repr__(self):
        return f'{self.__class__.__name__}(in_dim={self.in_dim}, mapping_size={self.mapping_size}, sigma={self.sigma})'


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
            b = sqrt(6 / self.in_f) / self.w0

        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-b, b)


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
                 final_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.blocks = nn.ModuleList()

        if self.num_layers < 1:
            raise ValueError(f'num_layers must be >= 1 (input to output); got {self.num_layers}')

        for i in range(self.num_layers):
            in_feat = self.in_dim if i == 0 else self.hidden_dim
            out_feat = self.out_dim if i + 1 == self.num_layers else self.hidden_dim

            is_first = i == 0
            is_last = i + 1 == self.num_layers

            curr_block = [block_factory(
                in_feat,
                out_feat,
                is_first=is_first,
                is_last=is_last
            )]
            if not is_last and dropout:
                curr_block.append(nn.Dropout(dropout))

            self.blocks.append(nn.Sequential(*curr_block))

        self.final_activation = final_activation
        if final_activation is None:
            self.final_activation = nn.Identity()

    def forward(self, x, modulations=None):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if modulations is not None and len(self.blocks) > i + 1:
                x *= modulations[i][:, None, None, :]
        return self.final_activation(x)


class ModulationNetwork(nn.Module):
    def __init__(self, in_dim: int, mod_dims: List[int], activation=nn.ReLU):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(len(mod_dims)):
            self.blocks.append(nn.Sequential(
                nn.Linear(in_dim + (mod_dims[i - 1] if i else 0), mod_dims[i]),
                activation()
            ))

    def forward(self, input):
        out = input
        mods = []
        for block in self.blocks:
            out = block(out)
            mods.append(out)
            out = torch.cat([out, input], dim=-1)
        return mods


class ImplicitDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 block_factory: BaseBlockFactory,
                 pos_encoder: CoordinateEncoding = None,
                 modulation: bool = False,
                 dropout: float = 0.0,
                 final_activation=torch.sigmoid):
        super().__init__()

        self.pos_encoder = pos_encoder
        self.latent_dim = latent_dim

        self.mod_network = None
        if modulation:
            self.mod_network = ModulationNetwork(
                in_dim=latent_dim,
                mod_dims=[hidden_dim for _ in range(num_layers - 1)],
                activation=nn.ReLU
            )

        self.net = MLP(
            in_dim=pos_encoder.out_dim + latent_dim * (not modulation),
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            block_factory=block_factory,
            dropout=dropout,
            final_activation=final_activation
        )

    def forward(self, input, latent):
        if self.pos_encoder is not None:
            input = self.pos_encoder(input)

        if self.mod_network is None:
            b, w, h, c = input.shape
            latent = latent[:, None, None, :].repeat(1, w, h, 1)
            out = self.net(torch.cat([latent, input], dim=-1))
        else:
            mods = self.mod_network(latent)
            out = self.net(input, mods)

        return out


class GON(nn.Module):
    def __init__(self, decoder: ImplicitDecoder, latent_updates: int = 1, learn_origin: bool = False):
        super().__init__()

        self.decoder = decoder
        self.latent_updates = latent_updates
        self.latent_updates = latent_updates

        if learn_origin:
            self.init_latent = nn.Parameter(torch.zeros(1, self.decoder.latent_dim))
        else:
            self.register_buffer('init_latent', torch.zeros(1, self.decoder.latent_dim))

    def get_init_latent(self, n):
        return self.init_latent.repeat(n, 1)

    def loss_inner(self, output, target):
        return F.binary_cross_entropy(
            output.view(-1), target.view(-1), reduction='none'
        ).view(target.shape).sum(0).mean()

    def loss_outer(self, output, target):
        return F.binary_cross_entropy(
            output.view(-1), target.view(-1), reduction='none'
        ).view(target.shape).mean()

    def infer_latents(self, input, target):
        latent = self.get_init_latent(len(target)).requires_grad_(True)

        for i in range(self.latent_updates):
            out = self.decoder(input, latent)
            inner_loss = self.loss_inner(out, target)
            latent = latent - torch.autograd.grad(inner_loss, [latent], create_graph=True, retain_graph=True)[0]

        return latent, inner_loss

    def forward(self, input, latent):
        return self.decoder(input, latent)
