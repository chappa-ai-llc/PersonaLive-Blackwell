import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .FAN_feature_extractor import FAN_SA
from einops import rearrange
from diffusers.models.modeling_utils import ModelMixin

def zero_module(module):
    # Zero out the parameters of a module and return it.
    assert isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear), type(module)
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sine-cosine position embedding of `pos`, as a NumPy array (M, D).

    The NumPy implementation diffusers shipped until 0.29; later releases take
    and return torch tensors instead.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class MotEncoder(ModelMixin):
    def __init__(self, out_ch=16):
        super().__init__()
        self.model = FAN_SA()
        self.out_drop = None #nn.Dropout(p=0.4)
        self.out_ch = out_ch
        expr_dim = 512
        extra_pos_embed = get_1d_sincos_pos_embed_from_grid(out_ch, np.arange(expr_dim//out_ch))
        self.register_buffer("pe", torch.from_numpy(extra_pos_embed).float().unsqueeze(0))
        self.final_proj = nn.Linear(expr_dim, expr_dim)
        self.out_bn = None

    def change_out_dim(self, out_ch):
        self.out_proj = nn.Linear(self.out_ch, out_ch)

    def set_attn_processor(self, processor):
        self.model.set_attn_processor(processor)

    def forward(self, x):
        x = x.to(self.dtype)
        latent = self.model(rearrange(x, "b c f h w -> (b f) c h w"))
        latent = self.final_proj(latent)
        latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
        latent = rearrange(latent, "(b f) l c -> b f l c", f=x.shape[2])
        return latent