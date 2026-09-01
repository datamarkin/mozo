# ------------------------------------------------------------------------
# BEN2 -- Background Erase Network
# Copyright (c) 2025 Prama LLC. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""``BEN_Base``: five feature maps in, one sigmoid matte out.

Extracted from ``BEN2.py`` lines 844-967. Three removals from ``forward``, none of them
arithmetic:

* **The two decorators are gone.** ``@torch.inference_mode()`` returns tensors that cannot be
  saved, re-entered or exported, and ``@torch.autocast(device_type="cuda", dtype=torch.float16)``
  silently halves the precision of the whole model on one class of machine. Both now live at the
  predictor boundary where they are visible and can be turned off, which is what makes this
  module gateable at all.
* **``if final_input == None`` is ``is None``.** Comparing a tensor to ``None`` with ``==`` works
  here only because the first iteration compares ``None`` to ``None``.
* **The ``inplace`` loop is gone.** Upstream walks every module setting ``m.inplace = True`` on
  ``nn.GELU`` and ``nn.Dropout``. ``nn.GELU`` has no such argument and ignores the attribute, and
  ``Dropout(inplace=)`` is inert in eval. Dead code that reads as load-bearing.

**Five convolutions are built and never run.** ``sideout1``..``sideout5`` are deep-supervision
heads from training: each is a ``Conv2d(128, 1, 3, padding=1)``, they carry 5,765 parameters
between them, and ``forward`` never mentions them. They are built anyway, because a strict load
needs somewhere to put their weights, and named here so a reader who greps for them finds a
recorded decision rather than a bug.
"""

from __future__ import annotations

__all__ = ["BEN_Base"]

import torch
import torch.nn as nn

from .blocks import MCLM, MCRM, image2patches, make_cbg, make_cbr, patches2image, rescale_to, resize_as
from .config import BACKBONE, DECODER
from .swin import SwinTransformer


class BEN_Base(nn.Module):
    """The published BEN2 model.

    Takes ``(N, 3, 1024, 1024)`` normalised with ImageNet statistics and returns
    ``(N, 1, 1024, 1024)`` of sigmoid opacity.

    The class name keeps upstream's spelling. It is what ``BEN2_Base.pth`` was saved from and
    what every issue thread and downstream integration calls it, and renaming it to
    ``Ben2Network`` would buy nothing but a broken grep.
    """

    def __init__(self) -> None:
        super().__init__()
        emb_dim = DECODER.emb_dim
        ch = BACKBONE.channels  # (128, 128, 256, 512, 1024)

        self.backbone = SwinTransformer(
            embed_dim=BACKBONE.embed_dim,
            depths=BACKBONE.depths,
            num_heads=BACKBONE.num_heads,
            window_size=BACKBONE.window_size,
            patch_size=BACKBONE.patch_size,
            drop_path_rate=BACKBONE.drop_path_rate,
            out_indices=BACKBONE.out_indices)

        # Deep-supervision heads. Built for the strict load, never called. See the module docstring.
        self.sideout5 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout4 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout3 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout2 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout1 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        # Deepest stage first: output5 takes stage 3's 1024 channels, output1 the patch embed's 128.
        self.output5 = make_cbr(ch[4], emb_dim)
        self.output4 = make_cbr(ch[3], emb_dim)
        self.output3 = make_cbr(ch[2], emb_dim)
        self.output2 = make_cbr(ch[1], emb_dim)
        self.output1 = make_cbr(ch[0], emb_dim)

        self.multifieldcrossatt = MCLM(emb_dim, DECODER.num_heads, DECODER.mclm_pools)
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = MCRM(emb_dim, DECODER.num_heads, DECODER.mcrm_pools)
        self.dec_blk2 = MCRM(emb_dim, DECODER.num_heads, DECODER.mcrm_pools)
        self.dec_blk3 = MCRM(emb_dim, DECODER.num_heads, DECODER.mcrm_pools)
        self.dec_blk4 = MCRM(emb_dim, DECODER.num_heads, DECODER.mcrm_pools)

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, DECODER.head_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(DECODER.head_width),
            nn.GELU(),
            nn.Conv2d(DECODER.head_width, DECODER.head_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(DECODER.head_width),
            nn.GELU(),
            nn.Conv2d(DECODER.head_width, emb_dim, kernel_size=3, padding=1))

        self.shallow = nn.Sequential(nn.Conv2d(3, emb_dim, kernel_size=3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim)
        self.upsample2 = make_cbg(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 3, 1024, 1024)`` -> ``(N, 1, 1024, 1024)``.

        One image costs **five** backbone forwards: the four 512x512 quadrants plus a 512x512
        bilinear downscale of the whole frame. That is the architecture, not a batching choice --
        the decoder splits its input ``[4, 1]`` at every rung and the two halves mean different
        things. An outer batch of *N* therefore sends ``5N`` images through the backbone at once,
        and since batched and unbatched reductions are not bit-identical, ``N`` is a number the
        gate has to pin rather than a free parameter.
        """
        real_batch = x.size(0)

        shallow_batch = self.shallow(x)
        glb_batch = rescale_to(x, scale_factor=0.5, interpolation="bilinear")

        final_input = None
        for i in range(real_batch):
            loc_batch = image2patches(x[i, :, :, :].unsqueeze(dim=0))
            input_ = torch.cat((loc_batch, glb_batch[i, :, :, :].unsqueeze(dim=0)), dim=0)
            final_input = input_ if final_input is None else torch.cat((final_input, input_), dim=0)

        features = self.backbone(final_input)

        outputs = []
        for i in range(real_batch):
            start, end = i * 5, (i + 1) * 5

            e5 = self.output5(features[4][start:end])
            e4 = self.output4(features[3][start:end])
            e3 = self.output3(features[2][start:end])
            e2 = self.output2(features[1][start:end])
            e1 = self.output1(features[0][start:end])

            loc_e5, glb_e5 = e5.split([4, 1], dim=0)
            e5 = self.multifieldcrossatt(loc_e5, glb_e5)

            # Each rung returns a saliency gate as well; upstream discards all four.
            e4, _ = self.dec_blk4(e4 + resize_as(e5, e4))
            e4 = self.conv4(e4)
            e3, _ = self.dec_blk3(e3 + resize_as(e4, e3))
            e3 = self.conv3(e3)
            e2, _ = self.dec_blk2(e2 + resize_as(e3, e2))
            e2 = self.conv2(e2)
            e1, _ = self.dec_blk1(e1 + resize_as(e2, e1))
            e1 = self.conv1(e1)

            loc_e1, glb_e1 = e1.split([4, 1], dim=0)
            output1_cat = patches2image(loc_e1)
            output1_cat = output1_cat + resize_as(glb_e1, output1_cat)

            final_output = self.insmask_head(output1_cat)
            shallow = shallow_batch[i, :, :, :].unsqueeze(dim=0)
            final_output = final_output + resize_as(shallow, final_output)
            # Both rescales take rescale_to's default 'nearest', while the resize_as calls above
            # are bilinear. Two filters on one path, and each call site keeps its own.
            final_output = self.upsample1(rescale_to(final_output))
            final_output = rescale_to(final_output + resize_as(shallow, final_output))
            final_output = self.upsample2(final_output)
            final_output = self.output(final_output)
            outputs.append(final_output.sigmoid())

        return torch.cat(outputs, dim=0)
