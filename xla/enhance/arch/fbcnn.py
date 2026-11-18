# Nguồn: https://github.com/jiaxi-jiang/FBCNN, file models/network_fbcnn.py (Apache License 2.0, xem LICENSE gốc).
# FBCNN — "Towards Flexible Blind JPEG Artifacts Removal", Jiang, Zhang, Timofte, ICCV 2021. Khối ``sequential``/``conv``
# lấy từ https://github.com/xinntao/BasicSR như ghi chú của file gốc.
#
# Copyright 2021 Jiaxi Jiang. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# Thay đổi của XLA (18/11/2026, theo mục 4(b) Apache 2.0): chỉ giữ các khối FBCNN dùng (conv, ResBlock, QFAttention,
# strideconv/convtranspose); bỏ import torchvision/numpy; đệm bằng phép số nguyên; định dạng lại cho ruff.
"""FBCNN: mạng khử artifact JPEG mù có dự đoán hệ số chất lượng (QF)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def sequential(*args: nn.Module) -> nn.Module:
    if len(args) == 1:
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(
    in_channels: int = 64,
    out_channels: int = 64,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    mode: str = "CBR",
) -> nn.Module:
    layers: list[nn.Module] = []
    for t in mode:
        if t == "C":
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == "T":
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == "R":
            layers.append(nn.ReLU(inplace=True))
        elif t == "r":
            layers.append(nn.ReLU(inplace=False))
        else:
            raise NotImplementedError(f"kiểu lớp không hỗ trợ: {t}")
    return sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, mode: str = "CRC") -> None:
        super().__init__()
        if mode[0] in ("R", "L"):
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, 3, 1, 1, True, mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res(x)


class QFAttention(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, mode: str = "CRC") -> None:
        super().__init__()
        if mode[0] in ("R", "L"):
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, 3, 1, 1, True, mode)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x + gamma * self.res(x) + beta


def _down(cin: int, cout: int) -> nn.Module:
    return conv(cin, cout, 2, 2, 0, True, "C")


def _up(cin: int, cout: int) -> nn.Module:
    return conv(cin, cout, 2, 2, 0, True, "T")


class FBCNN(nn.Module):
    def __init__(self, in_nc: int = 3, out_nc: int = 3, nc: tuple[int, ...] = (64, 128, 256, 512), nb: int = 4) -> None:
        super().__init__()
        act = "R"
        self.m_head = conv(in_nc, nc[0], bias=True, mode="C")
        self.nb = nb
        self.nc = nc
        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], "C" + act + "C") for _ in range(nb)], _down(nc[0], nc[1]))
        self.m_down2 = sequential(*[ResBlock(nc[1], nc[1], "C" + act + "C") for _ in range(nb)], _down(nc[1], nc[2]))
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], "C" + act + "C") for _ in range(nb)], _down(nc[2], nc[3]))
        self.m_body_encoder = sequential(*[ResBlock(nc[3], nc[3], "C" + act + "C") for _ in range(nb)])
        self.m_body_decoder = sequential(*[ResBlock(nc[3], nc[3], "C" + act + "C") for _ in range(nb)])
        self.m_up3 = nn.ModuleList([_up(nc[3], nc[2]), *[QFAttention(nc[2], nc[2], "C" + act + "C") for _ in range(nb)]])
        self.m_up2 = nn.ModuleList([_up(nc[2], nc[1]), *[QFAttention(nc[1], nc[1], "C" + act + "C") for _ in range(nb)]])
        self.m_up1 = nn.ModuleList([_up(nc[1], nc[0]), *[QFAttention(nc[0], nc[0], "C" + act + "C") for _ in range(nb)]])
        self.m_tail = conv(nc[0], out_nc, bias=True, mode="C")
        self.qf_pred = sequential(
            *[ResBlock(nc[3], nc[3], "C" + act + "C") for _ in range(nb)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.qf_embed = sequential(
            nn.Linear(1, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU()
        )
        self.to_gamma_3 = sequential(nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3 = sequential(nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2 = sequential(nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2 = sequential(nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1 = sequential(nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1 = sequential(nn.Linear(512, nc[0]), nn.Tanh())

    def forward(self, x: torch.Tensor, qf_input: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = x.size()[-2:]
        pad_b, pad_r = (-h) % 8, (-w) % 8
        if pad_b or pad_r:
            x = F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        qf = self.qf_pred(x)
        x = self.m_body_decoder(x)
        emb = self.qf_embed(qf_input) if qf_input is not None else self.qf_embed(qf)
        g3, b3 = self.to_gamma_3(emb), self.to_beta_3(emb)
        g2, b2 = self.to_gamma_2(emb), self.to_beta_2(emb)
        g1, b1 = self.to_gamma_1(emb), self.to_beta_1(emb)

        x = self.m_up3[0](x + x4)
        for i in range(self.nb):
            x = self.m_up3[i + 1](x, g3, b3)
        x = self.m_up2[0](x + x3)
        for i in range(self.nb):
            x = self.m_up2[i + 1](x, g2, b2)
        x = self.m_up1[0](x + x2)
        for i in range(self.nb):
            x = self.m_up1[i + 1](x, g1, b1)
        x = self.m_tail(x + x1)
        return x[..., :h, :w], qf
