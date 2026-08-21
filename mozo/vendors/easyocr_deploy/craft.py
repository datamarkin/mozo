"""CRAFT -- the detector. Character Region Awareness For Text detection.

A VGG16-BN encoder with a U-Net decoder and a two-channel head. The two channels are the whole
idea: one scores "this pixel is inside a character", the other "this pixel is between two
characters of the same word". Thresholding the first finds characters, thresholding the second
links them, and a connected component over the union is a word. No anchors, no proposals, no
NMS -- see :mod:`.boxes` for the half of that which is not a network.

Both maps come out at half the input resolution, which is why :func:`.boxes.rescale` starts by
multiplying by two.

Upstream builds the encoder by slicing ``torchvision.models.vgg16_bn().features`` at four
hard-coded indices. Here the layers are written out instead. The numbering is kept -- ``slice2``
starts at ``12`` because that is where torchvision's twelfth layer landed, and the published
checkpoint's keys say so -- but a slice index into a third-party model is exactly the kind of
thing that shifts silently under a version bump, and this package has to load one specific set
of weights forever.
"""

from __future__ import annotations

__all__ = ["CRAFT"]

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    """One 3x3 convolution with batch norm and ReLU, as torchvision's VGG stacks them."""
    return [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]


def _numbered(start: int, layers: list[nn.Module]) -> nn.Sequential:
    """``layers`` as a Sequential whose keys continue from ``start``.

    The checkpoint was written from ``add_module(str(i))`` over torchvision's own indices, so
    ``slice3``'s first layer is ``19`` and not ``0``. Renaming them at load time would be the
    alternative; keeping them means the state dict maps across with nothing but a prefix strip,
    and a key here can be grepped for in upstream unchanged.
    """
    return nn.Sequential(OrderedDict((str(start + i), layer) for i, layer in enumerate(layers)))


class VGG16BN(nn.Module):
    """The encoder, returning the five feature maps the decoder consumes.

    Note that the four ``slice`` boundaries fall *before* a ReLU rather than after one, so the
    tensors named ``relu*`` upstream are in fact post-batch-norm and pre-activation. The names
    are upstream's; the shapes are what matter, and the activation simply opens the next slice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.slice1 = _numbered(0, [
            *_conv_bn_relu(3, 64), *_conv_bn_relu(64, 64),
            nn.MaxPool2d(2, 2),
            *_conv_bn_relu(64, 128), *_conv_bn_relu(128, 128)[:2],
        ])
        self.slice2 = _numbered(12, [
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            *_conv_bn_relu(128, 256), *_conv_bn_relu(256, 256)[:2],
        ])
        self.slice3 = _numbered(19, [
            nn.ReLU(inplace=True), *_conv_bn_relu(256, 256),
            nn.MaxPool2d(2, 2),
            *_conv_bn_relu(256, 512), *_conv_bn_relu(512, 512)[:2],
        ])
        self.slice4 = _numbered(29, [
            nn.ReLU(inplace=True), *_conv_bn_relu(512, 512),
            nn.MaxPool2d(2, 2),
            *_conv_bn_relu(512, 512), *_conv_bn_relu(512, 512)[:2],
        ])
        # Upstream calls these fc6 and fc7 -- VGG's classifier head rewritten as convolutions,
        # dilated so the receptive field grows without another downsample. Built fresh rather
        # than slit off torchvision, so its numbering starts at zero and the max pool carries
        # no weights.
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Five maps, deepest first -- the order the decoder concatenates them in."""
        h = self.slice1(x)
        stage2 = h
        h = self.slice2(h)
        stage3 = h
        h = self.slice3(h)
        stage4 = h
        h = self.slice4(h)
        stage5 = h
        return self.slice5(h), stage5, stage4, stage3, stage2


class DoubleConv(nn.Module):
    """A 1x1 mix of the concatenated pair, then a 3x3 over the result."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CRAFT(nn.Module):
    """Image in, region and affinity maps out, both at half the input's resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.basenet = VGG16BN()
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, H/2, W/2, 2)`` -- region map in channel 0, affinity map in channel 1.

        Upstream also returns the 32-channel feature this head sits on. That output exists to
        feed CRAFT's optional refiner, a separate network for polygon-accurate boundaries that
        EasyOCR never instantiates and does not publish weights for, so it is dropped here.
        """
        sources = self.basenet(x)

        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        for upconv, source in ((self.upconv2, sources[2]), (self.upconv3, sources[3]),
                               (self.upconv4, sources[4])):
            y = F.interpolate(y, size=source.shape[2:], mode="bilinear", align_corners=False)
            y = upconv(torch.cat([y, source], dim=1))

        return self.conv_cls(y).permute(0, 2, 3, 1)
