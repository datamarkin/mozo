# SPDX-License-Identifier: Apache-2.0
"""The OWLv2 network: two CLIP towers and three heads, split at the seam that matters.

Upstream exposes one ``forward`` that takes an image *and* a prompt and runs both towers. That is
the wrong shape for serving. The two halves depend on different things -- the image tower on the
picture, the text tower on the phrase -- and in every real use one of them is held fixed while the
other varies. So this module publishes them separately:

    queries = model.encode_text(*tokenizer(["cat", "dog"]))   # once, for the whole corpus
    for picture in corpus:
        found = model.detect(model.encode_image(picture), queries, mask)

The saving is not marginal. The text tower is 12 blocks at 512 wide over 16 tokens; the image
tower is 12 blocks at 768 wide over 3,601. Running the prompt again for every picture is cheap in
absolute terms and pure waste, and upstream's ``image_text_embedder`` does exactly that. The same
seam is what would let each half be exported to a graph runtime on its own.

**Two tensors in the checkpoint are never run here**: ``visual_projection`` and ``logit_scale``.
They belong to CLIP's contrastive objective -- project the class token, dot it against the prompt,
scale by a learned temperature -- and the detector reads none of it. Upstream computes all three
on every call anyway, because it reaches the detection path through the full ``Owlv2Model``
forward. They are dropped rather than built. See ``PROVENANCE.md``.

**The class token is folded into every patch, multiplicatively.** Not concatenated, not dropped:
``patch * cls``, then a layer norm. It is how a per-patch feature gets told about the image as a
whole, and it is the one step here that is neither CLIP nor a plain detection head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import SPECS, Spec
from .heads import BoxHead, ClassHead, ObjectnessHead, box_bias
from .text.encoder import TextTower
from .vision.vit import VisionTower

__all__ = ["OwlV2"]


class OwlV2(nn.Module):
    """OWLv2 for a single image and a fixed set of phrases.

    Args:
        variant: Which published geometry. See :data:`~.config.SPECS`.

    Attributes:
        spec: The geometry in use.
        image_size: Square side the image tower runs at.
    """

    def __init__(self, variant: str = "base-ensemble"):
        super().__init__()
        if variant not in SPECS:
            raise ValueError(f"unknown variant {variant!r}; have {sorted(SPECS)}")
        self.spec: Spec = SPECS[variant]
        self.image_size = self.spec.vision.image_size

        self.vision = VisionTower(self.spec.vision)
        self.text = TextTower(self.spec.text)
        # No bias, which is CLIP's choice and the checkpoint's shape.
        self.text_projection = nn.Linear(
            self.spec.text.width, self.spec.text.projection, bias=False
        )

        width = self.spec.vision.width
        self.class_head = ClassHead(width, self.spec.text.projection)
        self.box_head = BoxHead(width)
        self.objectness_head = ObjectnessHead(width)
        self.layer_norm = nn.LayerNorm(width, eps=self.spec.vision.layer_norm_eps)

        # Constant once the geometry is fixed, and 3,600 rows of logarithms that would otherwise
        # be recomputed per image. Non-persistent so it moves with ``.to(device)`` and stays out
        # of the state dict -- it is derived from the geometry, not from the weights.
        self.register_buffer("box_bias", box_bias(self.spec.vision.patches), persistent=False)

    @torch.no_grad()
    def encode_text(self, ids: Tensor, mask: Tensor) -> Tensor:
        """Run the prompt half: tokenized phrases in, one embedding each out.

        Args:
            ids: ``(Q, L)`` token ids.
            mask: ``(Q, L)``, 1 where the row carries a real token.

        Returns:
            ``(Q, projection)``, L2-normalised. Cache this: it stays valid for every image.
        """
        queries = self.text_projection(self.text(ids, mask))
        return queries / torch.linalg.norm(queries, ord=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, pixels: Tensor) -> Tensor:
        """Run the expensive half: an image in, one feature per patch out.

        Args:
            pixels: ``(B, 3, image_size, image_size)`` normalised float tensor.

        Returns:
            ``(B, patches^2, width)``. Row-major over the grid, which is the order
            :func:`~.heads.box_bias` assumes and the order the boxes come back in.
        """
        sequence = self.vision(pixels)
        # ``(B, 1, W) * (B, P, W)`` broadcasts on its own. Upstream spells the class token's
        # expansion out against ``sequence[:, :-1].shape`` -- the *first* P tokens standing in for
        # the shape of the *last* P, which is the sort of thing that survives an off-by-one edit.
        return self.layer_norm(sequence[:, 1:] * sequence[:, :1])

    @torch.no_grad()
    def detect(
        self, patches: Tensor, queries: Tensor, query_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the heads: patch features and prompt embeddings in, raw predictions out.

        Args:
            patches: ``(B, P, width)`` from :meth:`encode_image`.
            queries: ``(Q, projection)`` from :meth:`encode_text`.
            query_mask: ``(Q,)`` bool. False marks a slot carrying no prompt.

        Returns:
            ``logits`` ``(B, P, Q)``, ``boxes`` ``(B, P, 4)`` as normalised cxcywh, and
            ``objectness`` ``(B, P)``. All three before any sigmoid or threshold -- what a
            detection *is* is the caller's decision, not this module's.
        """
        batch = patches.shape[0]
        logits = self.class_head(
            patches, queries.expand(batch, -1, -1), query_mask.expand(batch, -1)
        )
        return logits, self.box_head(patches, self.box_bias), self.objectness_head(patches)
