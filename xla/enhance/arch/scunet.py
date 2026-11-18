# Nguồn: https://github.com/cszn/SCUNet, file models/network_scunet.py (Apache License 2.0, xem LICENSE gốc).
# SCUNet — "Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis", Zhang et al., MIR 2023.
#
# Copyright 2022 Kai Zhang. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# Thay đổi của XLA (18/11/2026, theo mục 4(b) Apache 2.0): bỏ thop/einops/timm (không thêm gói): mọi ``rearrange`` viết
# lại bằng view/permute/reshape cùng thứ tự trục, ``trunc_normal_`` lấy từ torch, DropPath tự viết; bảng chỉ số
# relative position tính một lần thành buffer; bỏ ``print`` trong Block; masked_fill không in-place; định dạng lại.
"""SCUNet: khối Swin + conv song song trong UNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DropPath(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class WMSA(nn.Module):
    """Self-attention theo cửa sổ của Swin Transformer."""

    def __init__(self, input_dim: int, output_dim: int, head_dim: int, window_size: int, type: str) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        rp = torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        nn.init.trunc_normal_(rp, std=0.02)
        self.relative_position_params = nn.Parameter(
            rp.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1, 2).transpose(0, 1)
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        p = window_size
        cord = torch.stack(torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"), -1).reshape(-1, 2)
        relation = cord[:, None, :] - cord[None, :, :] + p - 1
        self.register_buffer("rel_idx0", relation[:, :, 0].long(), persistent=False)
        self.register_buffer("rel_idx1", relation[:, :, 1].long(), persistent=False)

    def generate_mask(self, h: int, w: int, p: int, shift: int, device: torch.device) -> torch.Tensor:
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=device)
        if self.type == "W":
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        # 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)'
        return attn_mask.reshape(1, 1, h * w, p * p, p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.window_size
        if self.type != "W":
            x = torch.roll(x, shifts=(-(p // 2), -(p // 2)), dims=(1, 2))
        b, hh, ww, c = x.shape
        w1, w2 = hh // p, ww // p
        # 'b (w1 p1) (w2 p2) c -> b (w1 w2) (p1 p2) c'
        x = x.view(b, w1, p, w2, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, w1 * w2, p * p, c)
        qkv = self.embedding_layer(x)
        # 'b nw np (threeh c) -> threeh b nw np c'
        qkv = qkv.view(b, w1 * w2, p * p, 3 * self.n_heads, self.head_dim).permute(3, 0, 1, 2, 4)
        q, k, v = qkv.chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        # 'h p q -> h 1 1 p q'
        sim = sim + self.relative_embedding().unsqueeze(1).unsqueeze(1)
        if self.type != "W":
            attn_mask = self.generate_mask(w1, w2, p, shift=p // 2, device=x.device)
            sim = sim.masked_fill(attn_mask, float("-inf"))
        probs = F.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        # 'h b w p c -> b w p (h c)'
        output = output.permute(1, 2, 3, 0, 4).reshape(b, w1 * w2, p * p, self.n_heads * self.head_dim)
        output = self.linear(output)
        # 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c'
        co = output.shape[-1]
        output = output.view(b, w1, w2, p, p, co).permute(0, 1, 3, 2, 4, 5).reshape(b, w1 * p, w2 * p, co)
        if self.type != "W":
            output = torch.roll(output, shifts=(p // 2, p // 2), dims=(1, 2))
        return output

    def relative_embedding(self) -> torch.Tensor:
        return self.relative_position_params[:, self.rel_idx0, self.rel_idx1]


class Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        type: str = "W",
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        assert type in ("W", "SW")
        self.type = "W" if input_resolution <= window_size else type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, 4 * input_dim), nn.GELU(), nn.Linear(4 * input_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.msa(self.ln1(x)))
        return x + self.drop_path(self.mlp(self.ln2(x)))


class ConvTransBlock(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        type: str = "W",
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        t = "W" if input_resolution <= window_size else type
        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, drop_path, t, input_resolution)
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


class SCUNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 3,
        config: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4),
        dim: int = 64,
        drop_path_rate: float = 0.0,
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        self.config = config
        self.dim = dim
        hd, ws = 32, 8
        self.window_size = ws
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        r = input_resolution

        def stage(cd: int, n: int, begin: int, res: int) -> list[nn.Module]:
            return [ConvTransBlock(cd, cd, hd, ws, dpr[i + begin], "W" if not i % 2 else "SW", res) for i in range(n)]

        begin = 0
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(*stage(dim // 2, config[0], begin, r), nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False))
        begin += config[0]
        self.m_down2 = nn.Sequential(
            *stage(dim, config[1], begin, r // 2), nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)
        )
        begin += config[1]
        self.m_down3 = nn.Sequential(
            *stage(2 * dim, config[2], begin, r // 4), nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)
        )
        begin += config[2]
        self.m_body = nn.Sequential(*stage(4 * dim, config[3], begin, r // 8))
        begin += config[3]
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), *stage(2 * dim, config[4], begin, r // 4)
        )
        begin += config[4]
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), *stage(dim, config[5], begin, r // 2)
        )
        begin += config[5]
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), *stage(dim // 2, config[6], begin, r)
        )
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        h, w = x0.size()[-2:]
        pad_b, pad_r = (-h) % 64, (-w) % 64
        if pad_b or pad_r:
            x0 = F.pad(x0, (0, pad_r, 0, pad_b), mode="replicate")
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :h, :w]
