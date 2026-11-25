# ------------------------------------------------------------------------
# BasicVSR++: chép từ MMEditing (https://github.com/open-mmlab/mmediting, nay mmagic; Apache License 2.0), file
# mmedit/models/backbones/sr_backbones/basicvsr_pp.py, basicvsr_net.py (SPyNet, ResidualBlocksWithInputConv),
# mmedit/models/common/{flow_warp.py, sr_backbone_utils.py (ResidualBlockNoBN), upsample.py (PixelShufflePack)}.
# (Chan et al., "BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment", CVPR 2022.)
# Trọng số dùng với file này: basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth (OpenMMLab, Apache-2.0).
# Thay đổi của XLA (25/11/2026): không mmcv (D8): ModulatedDeformConv2d của mmcv thay bằng torchvision.ops.deform_conv2d
# (cùng bố cục offset (dy, dx) theo vị trí kernel, có mask); ConvModule thay bằng Conv2d + ReLU giữ tên khoá
# ``basic_module.<i>.conv``; bỏ cpu_cache, kiểm mirror-extended, nhánh is_low_res_input=False, nạp spynet riêng, khởi
# tạo trọng số (chỉ nạp trọng số công bố); định dạng cho ruff.
#
# Apache License 2.0 — Copyright (c) OpenMMLab. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND.
# ------------------------------------------------------------------------
"""BasicVSR++ (×4, vào chuỗi LR [N, T, 3, H, W] RGB 0–1) với deformable conv của torchvision."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import deform_conv2d


def flow_warp(x: torch.Tensor, flow: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    """Dịch ``x`` [N, C, H, W] theo luồng ``flow`` [N, H, W, 2] (x, y), bilinear, align_corners=True."""
    _, _, h, w = x.size()
    gy, gx = torch.meshgrid(torch.arange(0, h, device=x.device), torch.arange(0, w, device=x.device), indexing="ij")
    grid = torch.stack((gx, gy), 2).type_as(x)
    gf = grid + flow
    gfx = 2.0 * gf[:, :, :, 0] / max(w - 1, 1) - 1.0
    gfy = 2.0 * gf[:, :, :, 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((gfx, gfy), dim=3), mode="bilinear", padding_mode=padding_mode,
                         align_corners=True)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(out_channels) for _ in range(num_blocks)]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, upsample_kernel: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, upsample_kernel,
                                       padding=(upsample_kernel - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class _ConvAct(nn.Module):  # ConvModule của mmcv: khoá ``conv.weight``/``conv.bias``
    def __init__(self, cin: int, cout: int, act: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 7, 1, 3)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SPyNetBasicModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(_ConvAct(8, 32, True), _ConvAct(32, 64, True), _ConvAct(64, 32, True),
                                          _ConvAct(32, 16, True), _ConvAct(16, 2, False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basic_module(x)


class SPyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.size()
        refs = [(ref - self.mean) / self.std]
        supps = [(supp - self.mean) / self.std]
        for _ in range(5):
            refs.append(F.avg_pool2d(refs[-1], kernel_size=2, stride=2, count_include_pad=False))
            supps.append(F.avg_pool2d(supps[-1], kernel_size=2, stride=2, count_include_pad=False))
        refs, supps = refs[::-1], supps[::-1]
        flow = refs[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(refs)):
            flow_up = flow if level == 0 else F.interpolate(flow, scale_factor=2, mode="bilinear",
                                                           align_corners=True) * 2.0
            warped = flow_warp(supps[level], flow_up.permute(0, 2, 3, 1), padding_mode="border")
            flow = flow_up + self.basic_module[level](torch.cat([refs[level], warped, flow_up], 1))
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        h, w = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        scale = torch.tensor([float(w) / float(w_up), float(h) / float(h_up)], device=flow.device, dtype=flow.dtype)
        return flow * scale.view(1, 2, 1, 1)


class SecondOrderDeformableAlignment(nn.Module):
    """Căn chỉnh deformable bậc hai (ModulatedDeformConv2d của mmcv → ``torchvision.ops.deform_conv2d``)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1,
                 deform_groups: int = 16, max_residue_magnitude: float = 10) -> None:
        super().__init__()
        self.padding = padding
        self.max_residue_magnitude = max_residue_magnitude
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels + 4, out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, extra_feat: torch.Tensor, flow_1: torch.Tensor,
                flow_2: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        return deform_conv2d(x, offset, self.weight, self.bias, padding=(self.padding, self.padding),
                             mask=torch.sigmoid(mask))


class BasicVSRPlusPlus(nn.Module):
    """Vào ``lqs`` [N, T, 3, H, W] RGB 0–1 (H, W ≥ 64), ra [N, T, 3, 4H, 4W]."""

    def __init__(self, mid_channels: int = 64, num_blocks: int = 7, max_residue_magnitude: float = 10) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for i, module in enumerate(("backward_1", "forward_1", "backward_2", "forward_2")):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3, padding=1, deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv((2 + i) * mid_channels, mid_channels, num_blocks)
        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def propagate(self, feats: dict[str, list[torch.Tensor]], flows: torch.Tensor,
                  module_name: str) -> dict[str, list[torch.Tensor]]:
        n, t, _, h, w = flows.size()
        frame_idx = list(range(t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ("spatial", module_name)] + [feat_prop]
            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=1))
            feats[module_name].append(feat_prop)
        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def upsample(self, lqs: torch.Tensor, feats: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        outputs = []
        mapping_idx = list(range(len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        for i in range(lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr = self.reconstruction(torch.cat(hr, dim=1))
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.conv_last(self.lrelu(self.conv_hr(hr)))
            outputs.append(hr + self.img_upsample(lqs[:, i, :, :, :]))
        return torch.stack(outputs, dim=1)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lqs.size()
        if h < 64 or w < 64:
            raise ValueError(f"cạnh ảnh vào phải ≥ 64, gặp {h}x{w}")
        feats: dict[str, list[torch.Tensor]] = {}
        f = self.feat_extract(lqs.view(-1, c, h, w))
        f = f.view(n, t, -1, f.shape[2], f.shape[3])
        feats["spatial"] = [f[:, i, :, :, :] for i in range(t)]
        flows_forward, flows_backward = self.compute_flow(lqs)
        for it in (1, 2):
            for direction in ("backward", "forward"):
                module = f"{direction}_{it}"
                feats[module] = []
                feats = self.propagate(feats, flows_backward if direction == "backward" else flows_forward, module)
        return self.upsample(lqs, feats)


def load_reds4(model: BasicVSRPlusPlus, path: str, strict: bool = True) -> BasicVSRPlusPlus:
    """Nạp checkpoint MMEditing (``state_dict`` khoá ``generator.*``)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    model.load_state_dict(sd, strict=strict)
    return model
