# ------------------------------------------------------------------------
# SwinIR: chép từ https://github.com/JingyunLiang/SwinIR (Apache License 2.0), file models/network_swinir.py
# (Jingyun Liang, "SwinIR: Image Restoration Using Swin Transformer", ICCVW 2021).
# Trọng số dùng với file này: 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth (SwinIR releases v0.0, Apache-2.0).
# Thay đổi của XLA (25/11/2026): chỉ giữ nhánh dùng cho SR ảnh thật (upsampler 'nearest+conv', resi_connection '1conv',
# patch_size 1, không ape, không use_checkpoint, không PatchMerging, không nhánh pixelshuffle/1conv-3conv, không FLOPs);
# DropPath/Dropout bỏ (chỉ suy luận + fine-tune không drop); timm.to_2tuple thay bằng tuple; mặt nạ cửa sổ dịch tính
# theo cỡ ảnh thật và nhớ đệm (không còn buffer ``attn_mask`` theo img_size 64: khoá đó bị bỏ khi nạp trọng số) để graph
# ONNX ảnh cố định chỉ có hằng số; ``mean`` thành buffer không lưu; định dạng cho ruff.
#
# Apache License 2.0 — Copyright (c) 2021 Jingyun Liang. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND.
# ------------------------------------------------------------------------
"""SwinIR (nhánh SR ảnh thật ×2/×4, upsampler 'nearest+conv')."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def window_reverse(windows: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / ws / ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        wh, ww = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * wh - 1) * (2 * ww - 1), num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(wh), torch.arange(ww)], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += wh - 1
        rel[:, :, 1] += ww - 1
        rel[:, :, 0] *= 2 * ww - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        wn = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(wn, wn, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


def calculate_mask(h: int, w: int, ws: int, ss: int) -> torch.Tensor:
    """Mặt nạ attention của cửa sổ dịch (SW-MSA) cho ảnh ``h``×``w``: 0 trong cùng vùng, −100 khác vùng."""
    img_mask = torch.zeros((1, h, w, 1))
    slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
    cnt = 0
    for hs in slices:
        for wsl in slices:
            img_mask[:, hs, wsl, :] = cnt
            cnt += 1
    mw = window_partition(img_mask, ws).view(-1, ws * ws)
    am = mw.unsqueeze(1) - mw.unsqueeze(2)
    return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: tuple[int, int], num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (self.window_size, self.window_size), num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self._masks: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def mask_for(self, h: int, w: int, ref: torch.Tensor) -> torch.Tensor:
        key = (h, w, str(ref.device), ref.dtype)
        if key not in self._masks:
            with torch.no_grad():
                self._masks[key] = calculate_mask(h, w, self.window_size, self.shift_size).to(ref.device, ref.dtype)
        return self._masks[key]

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape
        ws, ss = self.window_size, self.shift_size
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        shifted = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2)) if ss > 0 else x
        xw = window_partition(shifted, ws).view(-1, ws * ws, c)
        aw = self.attn(xw, mask=self.mask_for(h, w, x) if ss > 0 else None).view(-1, ws, ws, c)
        shifted = window_reverse(aw, ws, h, w)
        x = torch.roll(shifted, shifts=(ss, ss), dims=(1, 2)) if ss > 0 else shifted
        x = shortcut + x.reshape(b, h * w, c)
        return x + self.mlp(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(self, dim: int, input_resolution: tuple[int, int], depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float, qkv_bias: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size,
                                 0 if i % 2 == 0 else window_size // 2, mlp_ratio, qkv_bias)
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


def _embed(x: torch.Tensor) -> torch.Tensor:  # PatchEmbed patch_size 1: [B, C, H, W] → [B, HW, C]
    return x.flatten(2).transpose(1, 2)


def _unembed(x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:  # [B, HW, C] → [B, C, H, W]
    b, _, c = x.shape
    return x.transpose(1, 2).reshape(b, c, x_size[0], x_size[1])


class RSTB(nn.Module):
    def __init__(self, dim: int, input_resolution: tuple[int, int], depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float, qkv_bias: bool) -> None:
        super().__init__()
        self.residual_group = BasicLayer(dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return _embed(self.conv(_unembed(self.residual_group(x, x_size), x_size))) + x


class PatchEmbedNorm(nn.Module):
    """PatchEmbed (patch_size 1) có LayerNorm, giữ tên khoá ``patch_embed.norm`` của trọng số gốc."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(_embed(x))


class SwinIR(nn.Module):
    """SwinIR, nhánh ``nearest+conv`` (SR ảnh thật). Vào RGB 0–1 NCHW, ra ×``upscale``."""

    def __init__(self, upscale: int = 2, in_chans: int = 3, img_size: int = 64, window_size: int = 8,
                 img_range: float = 1.0, depths: Sequence[int] = (6, 6, 6, 6, 6, 6), embed_dim: int = 180,
                 num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6), mlp_ratio: float = 2.0, qkv_bias: bool = True,
                 upsampler: str = "nearest+conv", resi_connection: str = "1conv") -> None:
        super().__init__()
        if upsampler != "nearest+conv" or resi_connection != "1conv" or upscale not in (2, 4):
            raise ValueError("bản vendor chỉ giữ upsampler 'nearest+conv', resi_connection '1conv', upscale 2/4")
        num_feat = 64
        self.img_range = img_range
        self.upscale = upscale
        self.window_size = window_size
        self.register_buffer("mean", torch.tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1), persistent=False)
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        res = (img_size, img_size)
        self.patch_embed = PatchEmbedNorm(embed_dim)
        self.layers = nn.ModuleList(
            RSTB(embed_dim, res, d, h, window_size, mlp_ratio, qkv_bias) for d, h in zip(depths, num_heads, strict=True)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if upscale == 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (int(x.shape[2]), int(x.shape[3]))
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size)
        return _unembed(self.norm(x), x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = int(x.shape[2]), int(x.shape[3])
        ph = (self.window_size - h % self.window_size) % self.window_size
        pw = (self.window_size - w % self.window_size) % self.window_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), "reflect")
        mean = self.mean.to(x.dtype)
        x = (x - mean) * self.img_range
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
        if self.upscale == 4:
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        x = x / self.img_range + mean
        return x[:, :, : h * self.upscale, : w * self.upscale]
