"""A compact 2-D U-Net for multi-label brain-tumour segmentation.

Deliberately small (base width 24, ~2 M params) so it trains on CPU. Input is a
4-channel axial slice (FLAIR, T1w, T1gd, T2w); output is 3 sigmoid maps for the
nested BraTS regions WT / TC / ET.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, base=24):
        super().__init__()
        b = base
        self.d1 = DoubleConv(in_ch, b)
        self.d2 = DoubleConv(b, b * 2)
        self.d3 = DoubleConv(b * 2, b * 4)
        self.d4 = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(b * 8, b * 16)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.u4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.u3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.u2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.u1 = DoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        bn = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(bn), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x)          # logits (B, 3, H, W)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
